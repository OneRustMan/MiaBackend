// src/controllers/chat.controller.js
import { promises as fs } from "fs";

import { voice } from "../clients/elevenLabsClient.js";
import { ELEVEN_LABS_API_KEY, ELEVEN_LABS_MODEL_ID, MAX_USER_MSG_CHARS, VOICE_ID } from "../config/env.js";
import { audioFileToBase64, lipSyncMessage, readJsonTranscript } from "../services/audio.service.js";
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
      const idx = 0;
      const fileName = `audios/message_${idx}.mp3`;
      // Si textToSpeech falla sin lanzar, el guard de abajo debe ver que NO hay archivo:
      // sin este borrado, un mp3 de una corrida anterior pasaría el chequeo y se
      // devolvería audio viejo que no corresponde al mia_text actual.
      await fs.rm(fileName, { force: true }).catch(() => {});
      const { stability, similarityBoost } = getVoiceSettingsForEmotion(mia_emocion);

      const tTts = Date.now();
      await voice.textToSpeech(
        ELEVEN_LABS_API_KEY,
        VOICE_ID,
        fileName,
        mia_text,
        stability,
        similarityBoost,
        ELEVEN_LABS_MODEL_ID
      );
      logStep("voice.textToSpeech (ElevenLabs)", tTts);

      const stats = await fs.stat(fileName).catch(() => null);
      if (!stats || stats.size === 0) {
        throw new Error(`ElevenLabs no generó audio válido en ${fileName} (archivo inexistente o vacío). Revisa la API key, el voiceID o la cuota de ElevenLabs.`);
      }

      const tLipsync = Date.now();
      await lipSyncMessage(idx);
      logStep("lipSyncMessage (ffmpeg+rhubarb)", tLipsync);

      audio = await audioFileToBase64(fileName);
      lipsync = await readJsonTranscript(`audios/message_${idx}.json`);

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
