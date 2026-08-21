// src/controllers/realtimeChat.controller.js
//
// Sesión de chat en vivo por WebSocket (/chat/live).
//
// El cliente abre un socket, manda audio PCM crudo en frames binarios y, cuando
// termina de hablar, manda {"type":"commit"} para cerrar el turno. El audio va a
// la Realtime API (realtimeTranscription.service.js); cuando vuelve el transcript
// final, el turno sigue por el MISMO pipeline que usa el POST /chat de siempre
// (turnPipeline.service.js), así que la respuesta que recibe el cliente tiene el
// mismo shape de eventos que el SSE: chunk / done / aborted / error.
//
// Por qué el commit es manual: ninguno de los dos modelos de transcripción
// vigentes soporta VAD del servidor (ver realtimeTranscription.service.js), así
// que no hay corte de turno automático. Hoy lo dispara el script de prueba; en el
// frontend lo va a disparar la detección de silencio del lado del cliente.

import WebSocket from "ws";

import { OPENAI_REALTIME_TRANSCRIBE_MODEL } from "../config/env.js";
import { ensureDirs } from "../services/historial.service.js";
import { openRealtimeTranscriptionSession } from "../services/realtimeTranscription.service.js";
import { getSessionSnapshot } from "../services/session.service.js";
import { runTurnPipeline } from "../services/turnPipeline.service.js";
import { log } from "../utils/logger.js";

export function handleRealtimeChatConnection(ws) {
  // Mismo snapshot que hace el /chat SSE al arrancar un turno: si entra un
  // /reset, la generación avanza y esta conexión entera queda invalidada.
  const { generation: myGeneration, signal: mySignal } = getSessionSnapshot();

  let session = null;          // sesión de transcripción contra OpenAI
  let shuttingDown = false;
  let abortedSent = false;
  // Frames de audio que llegan mientras la sesión con OpenAI todavía se está
  // abriendo (el handshake tarda ~200-500ms y el cliente puede arrancar antes).
  const pendingAudio = [];
  // Los turnos se encadenan: si llega un commit mientras el anterior sigue
  // generando, el segundo espera en vez de pisarse con el primero.
  let turnChain = Promise.resolve();

  const send = (data) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
  };

  const onSessionAbort = () => {
    log(`⚠️ /chat/live: reset de sesión (gen ${myGeneration}), cerrando conexión.`);
    if (!abortedSent) {
      abortedSent = true;
      send({ type: "aborted" });
    }
    // Se corta primero con OpenAI (no dejar el socket colgado) y recién se cierra
    // con el cliente cuando el turno en vuelo terminó de avisar que abortó.
    closeUpstream();
    turnChain.finally(() => {
      if (ws.readyState === WebSocket.OPEN) ws.close(1000, "reset");
    });
  };

  function closeUpstream() {
    if (session) {
      session.close();
      session = null;
    }
  }

  function shutdown(reason) {
    if (shuttingDown) return;
    shuttingDown = true;
    mySignal.removeEventListener("abort", onSessionAbort);
    closeUpstream();
    log(`🔌 /chat/live cerrado (${reason}).`);
  }

  mySignal.addEventListener("abort", onSessionAbort, { once: true });

  ws.on("message", (data, isBinary) => {
    // Binario = chunk de audio PCM. Texto = mensaje de control.
    if (isBinary) {
      if (!session) { pendingAudio.push(data); return; }
      try {
        session.sendAudioChunk(data);
      } catch (err) {
        send({ type: "error", error: `No se pudo enviar audio: ${err.message}` });
      }
      return;
    }

    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      send({ type: "error", error: "Mensaje de control no es JSON válido." });
      return;
    }

    switch (msg.type) {
      case "commit":
        // Corte de turno manual (fase 3: lo dispara el silencio detectado en el front).
        if (!session) { send({ type: "error", error: "La sesión de transcripción todavía no está lista." }); return; }
        try {
          session.commitTurn();
          log("✂️  /chat/live: commit de turno recibido.");
        } catch (err) {
          send({ type: "error", error: `No se pudo cerrar el turno: ${err.message}` });
        }
        break;

      case "close":
        ws.close(1000, "cliente pidió cerrar");
        break;

      default:
        send({ type: "error", error: `Tipo de mensaje desconocido: ${msg.type}` });
    }
  });

  ws.on("close", () => shutdown("cliente desconectado"));
  ws.on("error", (err) => {
    console.error("Error en el WebSocket de /chat/live:", err);
    shutdown("error de socket");
  });

  // Arranque asíncrono: la sesión con OpenAI se abre mientras el cliente ya puede
  // estar mandando audio (por eso pendingAudio).
  (async () => {
    await ensureDirs();
    try {
      session = await openRealtimeTranscriptionSession({
        model: OPENAI_REALTIME_TRANSCRIBE_MODEL,
        onTranscript: (text) => queueTurn(text),
        onDelta: (delta) => send({ type: "transcript.delta", text: delta }),
        onError: (err) => {
          console.error("Error de la sesión de transcripción:", err);
          send({ type: "error", error: err.message });
        },
      });

      if (shuttingDown || mySignal.aborted) { closeUpstream(); return; }

      for (const chunk of pendingAudio) session.sendAudioChunk(chunk);
      pendingAudio.length = 0;

      log(`🎧 /chat/live listo (modelo ${OPENAI_REALTIME_TRANSCRIBE_MODEL}, sesión ${session.id}).`);
      send({ type: "ready", model: OPENAI_REALTIME_TRANSCRIBE_MODEL, sessionId: session.id });
    } catch (err) {
      console.error("No se pudo abrir la sesión de transcripción en vivo:", err);
      send({ type: "error", error: err.message });
      if (ws.readyState === WebSocket.OPEN) ws.close(1011, "transcripción no disponible");
    }
  })();

  function queueTurn(transcript) {
    turnChain = turnChain.then(() => runTurn(transcript)).catch((err) => {
      console.error("Error corriendo el turno de /chat/live:", err);
      send({ type: "error", error: err.message });
    });
  }

  async function runTurn(transcript) {
    const text = (transcript || "").trim();
    send({ type: "transcript", text });

    if (!text) {
      log("⚠️ /chat/live: transcripción vacía, no se genera turno.");
      send({ type: "skipped", reason: "transcripción vacía" });
      return;
    }

    log(`📝 /chat/live transcript: "${text}"`);
    const tTurn = Date.now();

    await runTurnPipeline({
      transcript: text,
      myGeneration,
      mySignal,
      startedAt: tTurn,
      label: "/chat/live",
      onChunk: (chunk) => send(chunk),
      onDone: () => send({ type: "done" }),
      onAborted: () => {
        if (abortedSent) return;
        abortedSent = true;
        send({ type: "aborted" });
      },
      // A diferencia del SSE, acá el evento de error se manda siempre: el socket
      // queda abierto para el turno siguiente, así que el cliente necesita un
      // terminador aunque ya hayan salido chunks.
      onError: (err) => send({ type: "error", error: err.message }),
    });
  }
}
