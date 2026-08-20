// src/controllers/chat.controller.js
import OpenAI from "openai";

import { MAX_USER_MSG_CHARS } from "../config/env.js";
import { generateSpeechWithTimestamps } from "../services/audio.service.js";
import { broadcastTurn } from "../services/dashboard.service.js";
import {
  callLocalMiaPredict,
  callLocalSentiment,
  getVoiceSettingsForEmotion,
  mapEmotionToVisuals,
} from "../services/emotion.service.js";
import { GREETING_TEXT, getGreetingAudio } from "../services/greeting.service.js";
import { ensureDirs, readHistorial, writeHistorial } from "../services/historial.service.js";
import { generateMiaReplyStream } from "../services/mia.service.js";
import { getSessionSnapshot } from "../services/session.service.js";
import {
  condenseUserMessageIfNeeded,
  parseDataUrl,
  transcribeBufferWithWhisper,
} from "../services/transcription.service.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { log } from "../utils/logger.js";
import { withFallback } from "../utils/withFallback.js";

// Instrumentación temporal: mide cada paso secuencial de la rama de audio
// para saber dónde se va el tiempo del turno. No altera el flujo.
function logStep(label, startedAt) {
  log(`⏱️  ${label}: ${Date.now() - startedAt}ms`);
}

export const handleChat = asyncHandler(async (req, res) => {
  const tChatStart = Date.now();
  // Snapshot de la sesión al arrancar: si llega un /reset mientras trabajamos,
  // la generación avanza y sabemos que este turno ya no sirve.
  const { generation: myGeneration, signal: mySignal } = getSessionSnapshot();
  // Se deja en req para que errorHandler pueda loguear la generación al abortar.
  req.sessionGeneration = myGeneration;

  await ensureDirs();
  const userMessage = req.body.message;

  // Tipo inválido (número, objeto, booleano...) es distinto de "no hay mensaje":
  // se corta acá para que más abajo .trim() tenga garantizado un string.
  if (userMessage !== undefined && userMessage !== null && typeof userMessage !== "string") {
    return res.status(400).json({
      ok: false,
      error: "El campo 'message' debe ser un string.",
      messages: [],
    });
  }

  if (typeof userMessage === "string" && userMessage.startsWith("data:audio")) {
    log("🎙️ Audio recibido, transcribiendo...");
    const { buffer, mime, ext } = parseDataUrl(userMessage);
    const tWhisper = Date.now();
    let transcript = await transcribeBufferWithWhisper(buffer, `audio.${ext}`, mime, mySignal);
    logStep("transcribeBufferWithWhisper", tWhisper);

    // Mismo criterio que usa condenseUserMessageIfNeeded internamente, solo para
    // saber si el paso realmente corrió o se saltó (no cambia el llamado).
    const condenseWillRun = Boolean(transcript) && transcript.length > MAX_USER_MSG_CHARS;
    const tCondense = Date.now();
    transcript = await condenseUserMessageIfNeeded(transcript, mySignal);
    if (condenseWillRun) logStep("condenseUserMessageIfNeeded", tCondense);
    else log("⏱️  condenseUserMessageIfNeeded: skipped");

    const tSentiment = Date.now();
    const sentimiento = await withFallback(
      () => callLocalSentiment(transcript, mySignal), "neutral", "sentiment");
    logStep("callLocalSentiment", tSentiment);

    const tPredict = Date.now();
    const mia_emocion = await withFallback(
      () => callLocalMiaPredict(transcript, sentimiento, mySignal), "default", "mia_predict");
    logStep("callLocalMiaPredict", tPredict);

    const historialActual = await readHistorial();
    const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
    const nextKey = `conversacion_${nextIndex}`;

    const visuals = mapEmotionToVisuals(mia_emocion, nextIndex - 1);

    // A partir de acá la respuesta deja de ser un JSON: cada frase sale por SSE
    // apenas tiene su audio listo, así MIA empieza a sonar sin esperar el texto
    // completo. Ya no hay res.json() en el camino feliz de esta rama.
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });

    const { stability, similarityBoost } = getVoiceSettingsForEmotion(mia_emocion);
    let fullText = "";
    let sentAny = false;
    let chunkCount = 0;
    const tStream = Date.now();

    try {
      for await (const sentence of generateMiaReplyStream({ transcript, sentimiento, mia_emocion, turnIndex: nextIndex, signal: mySignal })) {
        // Cuánto tardó el modelo en soltar la PRIMERA frase completa: es el
        // techo real del "MIA empieza a hablar rápido", antes de sumarle TTS.
        if (!sentAny) logStep("primera frase del modelo (sin TTS)", tStream);
        fullText += (fullText ? " " : "") + sentence;

        let chunkAudio;
        let chunkLipsync;
        try {
          const tTts = Date.now();
          const result = await generateSpeechWithTimestamps(sentence, stability, similarityBoost, mySignal);
          chunkAudio = result.audio;
          chunkLipsync = result.lipsync;
          logStep(`generateSpeechWithTimestamps (chunk ${chunkCount + 1})`, tTts);
        } catch (ttsErr) {
          if (ttsErr.name === "AbortError") throw ttsErr;
          console.error("Error generando audio de un chunk:", ttsErr);
          // Degradación elegante: mandamos el texto del chunk igual, sin audio.
        }

        res.write(`data: ${JSON.stringify({
          type: "chunk",
          text: sentence,
          audio: chunkAudio,
          lipsync: chunkLipsync,
          facialExpression: visuals.facialExpression,
          animation: visuals.animation,
        })}\n\n`);
        chunkCount += 1;
        // El número que importa para "MIA empieza a hablar rápido" es este,
        // no el total del turno.
        if (!sentAny) logStep("PRIMER CHUNK (texto+audio en el aire)", tChatStart);
        sentAny = true;
      }

      // El SDK de OpenAI no lanza cuando se aborta el fetch a mitad del
      // stream: corta la iteración en silencio y el for-await termina normal.
      // Sin este chequeo, un /reset en vuelo le mandaría un "done" al frontend
      // como si el turno se hubiera completado.
      if (mySignal.aborted) {
        const abortErr = new Error("Stream abortado por reset");
        abortErr.name = "AbortError";
        throw abortErr;
      }

      // El historial se escribe recién acá porque el mia_text completo no
      // existe hasta que el stream termina.
      if (myGeneration === getSessionSnapshot().generation) {
        const historialFinal = await readHistorial();
        historialFinal[nextKey] = { user_responde: transcript, sentimiento, mia_emocion, mia_text: fullText };
        const tWrite = Date.now();
        await writeHistorial(historialFinal);
        logStep("writeHistorial", tWrite);
        broadcastTurn({ type: "turn", transcript, sentimiento, mia_emocion, mia_text: fullText });
        log(`💾 Turno guardado como ${nextKey} (${chunkCount} chunks).`);
      } else {
        log(`⚠️ Sesión reseteada durante ${nextKey}; se descarta la escritura.`);
      }

      res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
    } catch (err) {
      if (err.name === "AbortError" || err instanceof OpenAI.APIUserAbortError) {
        log(`⚠️ Stream cancelado por reset (gen ${myGeneration}).`);
        res.write(`data: ${JSON.stringify({ type: "aborted" })}\n\n`);
      } else {
        console.error("Error en stream de /chat:", err);
        if (!sentAny) {
          res.write(`data: ${JSON.stringify({ type: "error", error: err.message })}\n\n`);
        }
        // Si ya mandamos chunks, no hay forma limpia de "fallar" a mitad de
        // camino: el frontend ya recibió audio parcial, cerramos el stream.
      }
    }

    logStep(`TOTAL /chat (${chunkCount} chunks)`, tChatStart);
    res.end();
    return;
  }

  // Texto sin audio → saludo pregenerado desde assets/, 100% desde memoria
  // (sin ElevenLabs, sin rhubarb y sin leer historial.json: el saludo no se guarda).
  if (!userMessage || !userMessage.trim()) {
    try {
      const { audio, lipsync } = await getGreetingAudio();
      return res.json({ ok: true, messages: [{ text: GREETING_TEXT, audio, lipsync, facialExpression: "default", animation: "Talking_0" }] });
    } catch (greetingErr) {
      console.error("Error sirviendo el saludo local:", greetingErr);
      return res.json({ ok: true, messages: [{ text: GREETING_TEXT, facialExpression: "default", animation: "Talking_0" }] });
    }
  }

  // Cualquier texto que no sea audio y no esté vacío es un uso
  // indebido de la API: este backend solo acepta audio (data:audio...)
  // o mensaje vacío (saludo). No lo procesamos con IA porque no pasaría
  // por el pipeline de sentimiento/historial ni por los guards
  // de cancelación de sesión que sí respetan las otras dos ramas.
  return res.status(400).json({
    ok: false,
    error: "Este backend solo acepta audio o mensaje vacío (saludo).",
    messages: [],
  });
});
