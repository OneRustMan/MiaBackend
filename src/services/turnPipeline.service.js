// src/services/turnPipeline.service.js
//
// Pipeline de un turno de MIA, desde que existe el transcript hasta que el turno
// quedó guardado: sentimiento → emoción → stream de frases → TTS por frase →
// historial → broadcast al dashboard.
//
// Vive acá y no en el controller porque hay DOS transportes que lo usan igual:
//   - POST /chat        (SSE, chat.controller.js)          → res.write("data: ...")
//   - WS   /chat/live   (realtimeChat.controller.js)       → ws.send(...)
// El cuerpo es exactamente el que vivía en la rama de audio de chat.controller.js;
// lo único que cambió es que en vez de escribir en `res` avisa por callbacks.
//
// NO incluye la transcripción: quién llama ya tiene el texto (Whisper en el SSE,
// Realtime API en el WebSocket). Ese es justamente el punto de corte que permite
// compartir todo lo de abajo.

import OpenAI from "openai";

import { generateSpeechWithTimestamps } from "./audio.service.js";
import { broadcastTurn } from "./dashboard.service.js";
import {
  callLocalMiaPredict,
  callLocalSentiment,
  getVoiceSettingsForEmotion,
  mapEmotionToVisuals,
} from "./emotion.service.js";
import { readHistorial, writeHistorial } from "./historial.service.js";
import { generateMiaReplyStream } from "./mia.service.js";
import { getSessionSnapshot } from "./session.service.js";
import { log, logStep } from "../utils/logger.js";
import { withFallback } from "../utils/withFallback.js";

/**
 * Corre un turno completo y va emitiendo por callbacks.
 *
 * Los errores se reparten en dos grupos, igual que antes del refactor:
 *   - Lo previo al stream (sentimiento, emoción, lectura de historial) LANZA hacia
 *     arriba, para que en el SSE lo siga manejando asyncHandler → errorHandler
 *     (respuesta JSON, todavía no se mandaron headers de stream).
 *   - Lo de adentro del stream se reporta por onAborted()/onError() y NO lanza,
 *     porque a esa altura el transporte ya está abierto y no hay forma limpia de
 *     "fallar" a mitad de camino.
 *
 * @param {object}   opts
 * @param {string}   opts.transcript     Texto del usuario, ya transcripto.
 * @param {number}   opts.myGeneration   Generación de sesión al arrancar el turno.
 * @param {AbortSignal} opts.mySignal    Señal de esa misma generación.
 * @param {number}   [opts.startedAt]    Timestamp del inicio del turno (para el TOTAL).
 * @param {string}   [opts.label]        Etiqueta del transporte, solo para logs.
 * @param {Function} [opts.onStart]      Se llama justo antes de empezar a generar,
 *                                       cuando ya no puede haber respuesta JSON de error.
 *                                       En SSE es donde se escriben los headers del stream.
 * @param {Function} [opts.onChunk]      (chunkData) por cada frase con su audio.
 * @param {Function} [opts.onDone]       Turno completo y guardado.
 * @param {Function} [opts.onAborted]    Turno cancelado por /reset.
 * @param {Function} [opts.onError]      (err, { sentAny }) error real del stream.
 */
export async function runTurnPipeline({
  transcript,
  myGeneration,
  mySignal,
  startedAt = Date.now(),
  label = "/chat",
  onStart,
  onChunk,
  onDone,
  onAborted,
  onError,
}) {
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

  // A partir de acá la respuesta deja de poder ser un JSON de error: cada frase
  // sale apenas tiene su audio listo, así MIA empieza a sonar sin esperar el
  // texto completo. En el SSE este es el punto donde se escriben los headers.
  await onStart?.();

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

      await onChunk?.({
        type: "chunk",
        text: sentence,
        audio: chunkAudio,
        lipsync: chunkLipsync,
        facialExpression: visuals.facialExpression,
        animation: visuals.animation,
      });
      chunkCount += 1;
      // El número que importa para "MIA empieza a hablar rápido" es este,
      // no el total del turno.
      if (!sentAny) logStep("PRIMER CHUNK (texto+audio en el aire)", startedAt);
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

    await onDone?.();
  } catch (err) {
    if (err.name === "AbortError" || err instanceof OpenAI.APIUserAbortError) {
      log(`⚠️ Stream cancelado por reset (gen ${myGeneration}).`);
      await onAborted?.();
    } else {
      console.error(`Error en stream de ${label}:`, err);
      // El guard de sentAny lo decide el transporte: si ya salieron chunks, el
      // cliente recibió audio parcial y el SSE prefiere cerrar sin evento de error.
      await onError?.(err, { sentAny });
    }
  }

  logStep(`TOTAL ${label} (${chunkCount} chunks)`, startedAt);

  return { transcript, sentimiento, mia_emocion, fullText, chunkCount, sentAny };
}

