// src/controllers/chat.controller.js
import { promises as fs } from "fs";

import { voice } from "../clients/elevenLabsClient.js";
import { openai } from "../clients/openaiClient.js";
import { ELEVEN_LABS_API_KEY, VOICE_ID } from "../config/env.js";
import { audioFileToBase64, lipSyncMessage, readJsonTranscript } from "../services/audio.service.js";
import {
  callLocalMiaPredict,
  callLocalSentiment,
  getVoiceSettingsForEmotion,
  mapEmotionToVisuals,
} from "../services/emotion.service.js";
import { ensureDirs, readHistorial, writeHistorial } from "../services/historial.service.js";
import { generateMiaReply, updateRollingSummary } from "../services/mia.service.js";
import { getSessionSnapshot } from "../services/session.service.js";
import {
  condenseUserMessageIfNeeded,
  parseDataUrl,
  transcribeBufferWithWhisper,
} from "../services/transcription.service.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { log } from "../utils/logger.js";
import { withFallback } from "../utils/withFallback.js";

export const handleChat = asyncHandler(async (req, res) => {
  // Snapshot de la sesión al arrancar: si llega un /reset mientras trabajamos,
  // la generación avanza y sabemos que este turno ya no sirve.
  const { generation: myGeneration, signal: mySignal } = getSessionSnapshot();
  // Se deja en req para que errorHandler pueda loguear la generación al abortar.
  req.sessionGeneration = myGeneration;

  await ensureDirs();
  const userMessage = req.body.message;

  if (typeof userMessage === "string" && userMessage.startsWith("data:audio")) {
    log("🎙️ Audio recibido, transcribiendo...");
    const { buffer, mime, ext } = parseDataUrl(userMessage);
    let transcript = await transcribeBufferWithWhisper(buffer, `audio.${ext}`, mime, mySignal);

    transcript = await condenseUserMessageIfNeeded(transcript, mySignal);

    const sentimiento = await withFallback(
      () => callLocalSentiment(transcript, mySignal), "neutral", "sentiment");
    const mia_emocion = await withFallback(
      () => callLocalMiaPredict(transcript, sentimiento, mySignal), "default", "mia_predict");

    const historialActual = await readHistorial();
    const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
    const nextKey = `conversacion_${nextIndex}`;

    const mia_text = await generateMiaReply({ transcript, sentimiento, mia_emocion, signal: mySignal });
    const visuals = mapEmotionToVisuals(mia_emocion, nextIndex - 1);

    // Guardamos el texto ANTES de tocar audio: si ElevenLabs falla, el turno no se pierde.
    historialActual[nextKey] = { user_responde: transcript, sentimiento, mia_emocion, mia_text };
    if (myGeneration === getSessionSnapshot().generation) {
      await writeHistorial(historialActual);
      await updateRollingSummary(historialActual[nextKey], mySignal);
    } else {
      log(`⚠️ Sesión reseteada durante ${nextKey}; se descarta la escritura.`);
    }

    log(`💾 Turno guardado como ${nextKey}, generando audio...`);

    if (myGeneration !== getSessionSnapshot().generation) {
      log(`⚠️ Sesión reseteada antes de generar audio para ${nextKey}; se omite ElevenLabs.`);
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
      const { stability, similarityBoost } = getVoiceSettingsForEmotion(mia_emocion);

      await voice.textToSpeech(
        ELEVEN_LABS_API_KEY,
        VOICE_ID,
        fileName,
        mia_text,
        stability,
        similarityBoost,
        "eleven_multilingual_v2"
      );

      const stats = await fs.stat(fileName).catch(() => null);
      if (!stats || stats.size === 0) {
        throw new Error(`ElevenLabs no generó audio válido en ${fileName} (archivo inexistente o vacío). Revisa la API key, el voiceID o la cuota de ElevenLabs.`);
      }

      await lipSyncMessage(idx);
      audio = await audioFileToBase64(fileName);
      lipsync = await readJsonTranscript(`audios/message_${idx}.json`);

      log(`✅ Audio flujo completo generado para ${nextKey}`);
    } catch (audioErr) {
      console.error("Error generando audio/lipsync:", audioErr);
    }

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

  // Texto sin audio → saludo on-the-fly (sin intro_*.wav)
  if (!userMessage) {
    const mia_text = "Estoy aquí. ¿Qué te gustaría contarme?";
    const historialActual = await readHistorial();
    const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
    const visuals = mapEmotionToVisuals("default", nextIndex - 1);

    try {
      const idx = 0;
      const fileName = `audios/message_${idx}.mp3`;
      await voice.textToSpeech(
        ELEVEN_LABS_API_KEY,
        VOICE_ID,
        fileName,
        mia_text,
        0.5,
        0.75,
        "eleven_multilingual_v2"
      );
      await lipSyncMessage(idx);
      const audio = await audioFileToBase64(fileName);
      const lipsync = await readJsonTranscript(`audios/message_${idx}.json`);
      return res.json({ ok: true, messages: [{ text: mia_text, audio, lipsync, facialExpression: visuals.facialExpression, animation: visuals.animation }] });
    } catch {
      return res.json({ ok: true, messages: [{ text: mia_text, facialExpression: visuals.facialExpression, animation: visuals.animation }] });
    }
  }

  // (Tu rama vieja de chat JSON con 3.5 se mantiene por compat)
  const completion = await openai.chat.completions.create({
    model: "gpt-3.5-turbo-1106",
    max_tokens: 1000,
    temperature: 0.6,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: `
          You are a virtual girlfriend.
          You will always reply with a JSON array of messages. With a maximum of 3 messages.
          Each message has a text, facialExpression, and animation property.
          The animations allowed are only Talking_0, Talking_1, Talking_2.
        `},
      { role: "user", content: userMessage },
    ],
  });

  let messages = JSON.parse(completion.choices[0].message.content);
  if (messages.messages) messages = messages.messages;

  await fs.mkdir("audios", { recursive: true });
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const fileName = `audios/message_${i}.mp3`;
    await voice.textToSpeech(
      ELEVEN_LABS_API_KEY,
      VOICE_ID,
      fileName,
      m.text,
      0.5,
      0.75,
      "eleven_multilingual_v2"
    );
    await lipSyncMessage(i);
    m.audio = await audioFileToBase64(fileName);
    m.lipsync = await readJsonTranscript(`audios/message_${i}.json`);
  }

  res.send({ messages });
});
