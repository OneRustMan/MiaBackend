// src/controllers/chat.controller.js
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
import { generateMiaReply } from "../services/mia.service.js";
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

    const tReply = Date.now();
    const mia_text = await generateMiaReply({ transcript, sentimiento, mia_emocion, turnIndex: nextIndex, signal: mySignal });
    logStep("generateMiaReply", tReply);

    const visuals = mapEmotionToVisuals(mia_emocion, nextIndex - 1);

    // Espejo en vivo para el dashboard de la feria; síncrono, no bloquea el turno.
    broadcastTurn({ type: "turn", transcript, sentimiento, mia_emocion, mia_text });

    // Guardamos el texto ANTES de tocar audio: si ElevenLabs falla, el turno no se pierde.
    historialActual[nextKey] = { user_responde: transcript, sentimiento, mia_emocion, mia_text };
    if (myGeneration === getSessionSnapshot().generation) {
      const tWrite = Date.now();
      await writeHistorial(historialActual);
      logStep("writeHistorial", tWrite);
    } else {
      log(`⚠️ Sesión reseteada durante ${nextKey}; se descarta la escritura.`);
    }

    log(`💾 Turno guardado como ${nextKey}, generando audio...`);

    if (myGeneration !== getSessionSnapshot().generation) {
      log(`⚠️ Sesión reseteada antes de generar audio para ${nextKey}; se omite ElevenLabs.`);
      logStep("TOTAL /chat", tChatStart);
      return res.json({
        ok: true,
        transcript,
        sentimiento,
        mia_emocion,
        messages: [
          {
            text: mia_text,
            facialExpression: visuals.facialExpression,
            animation: visuals.animation,
          },
        ],
      });
    }

    let audio;
    let lipsync;
    try {
      const { stability, similarityBoost } = getVoiceSettingsForEmotion(mia_emocion);

      // Audio + timing de cada carácter en una sola respuesta: sin archivos
      // intermedios, sin ffmpeg y sin rhubarb (por eso tampoco hace falta el
      // guard de archivo reciclado que había acá).
      const tTts = Date.now();
      ({ audio, lipsync } = await generateSpeechWithTimestamps(mia_text, stability, similarityBoost, mySignal));
      logStep("generateSpeechWithTimestamps (ElevenLabs with-timestamps)", tTts);

      log(`✅ Audio flujo completo generado para ${nextKey}`);
    } catch (audioErr) {
      console.error("Error generando audio/lipsync:", audioErr);
    }

    logStep("TOTAL /chat", tChatStart);
    return res.json({
      ok: true,
      transcript,
      sentimiento,
      mia_emocion,
      messages: [
        {
          text: mia_text,
          audio,
          lipsync,
          facialExpression: visuals.facialExpression,
          animation: visuals.animation,
        },
      ],
    });
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
