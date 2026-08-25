// src/controllers/chat.controller.js
import { MAX_USER_MSG_CHARS } from "../config/env.js";
import { GREETING_TEXT, getGreetingAudio } from "../services/greeting.service.js";
import { ensureDirs } from "../services/historial.service.js";
import { getSessionSnapshot } from "../services/session.service.js";
import {
  condenseUserMessageIfNeeded,
  parseDataUrl,
  transcribeBufferWithWhisper,
} from "../services/transcription.service.js";
import { runTurnPipeline } from "../services/turnPipeline.service.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { log, logStep } from "../utils/logger.js";

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

    // De acá para abajo el turno es idéntico al de /chat/live: el pipeline es
    // compartido y este handler solo lo traduce a eventos SSE.
    const sse = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

    await runTurnPipeline({
      transcript,
      myGeneration,
      mySignal,
      startedAt: tChatStart,
      label: "/chat",
      // A partir de acá la respuesta deja de ser un JSON: cada frase sale por SSE
      // apenas tiene su audio listo, así MIA empieza a sonar sin esperar el texto
      // completo. Ya no hay res.json() en el camino feliz de esta rama.
      onStart: () => res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      }),
      onChunk: (chunk) => sse(chunk),
      onDone: () => sse({ type: "done" }),
      onAborted: () => sse({ type: "aborted" }),
      onError: (err, { sentAny }) => {
        // Si ya mandamos chunks, no hay forma limpia de "fallar" a mitad de
        // camino: el frontend ya recibió audio parcial, cerramos el stream.
        if (!sentAny) sse({ type: "error", error: err.message });
      },
    });

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
