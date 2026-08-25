// src/services/realtimeTranscription.service.js
//
// Sesión de transcripción en vivo contra la Realtime API de OpenAI (interfaz GA).
//
// Por qué WebSocket "a mano" y no el helper del SDK:
//   openai@6.x trae OpenAIRealtimeWS (openai/realtime/ws), pero su buildRealtimeURL()
//   arma SIEMPRE la URL como /v1/realtime?model=<modelo> y no permite inyectar
//   ?intent=transcription. La API rechaza un modelo de transcripción en ?model=
//   ("Model X is a transcription model and cannot be used as the realtime session model"),
//   así que el helper nativo no sirve para abrir una sesión de tipo "transcription".
//   Usamos `ws` directo, que es además lo que muestra la doc oficial.
//
// Forma confirmada contra la API real (ver scripts/test-realtime-transcription.js):
//   URL     : wss://api.openai.com/v1/realtime?intent=transcription
//   Headers : Authorization: Bearer <API_KEY>   (ya NO hace falta OpenAI-Beta)
//   Config  : evento session.update con session.type = "transcription"
//   Turno   : MANUAL en los dos modelos vigentes. Ni gpt-realtime-whisper ni
//             gpt-live-transcribe soportan VAD del servidor (rechazan turn_detection),
//             así que se manda turn_detection: null y el corte lo hace quien use este
//             servicio llamando a commitTurn(). No hay corte automático de turno.
//             Si algún día se suma un modelo con server_vad real (p. ej. gpt-transcribe),
//             ahí sí correspondería documentar ese camino; hoy no aplica.
//   Final   : conversation.item.input_audio_transcription.completed  -> event.transcript

import WebSocket from "ws";
import { OPENAI_API_KEY, OPENAI_REALTIME_TRANSCRIBE_MODEL } from "../config/env.js";

export const REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription";

// La API espera PCM 16-bit little-endian, mono. Solo se soporta 24 kHz.
export const PCM_SAMPLE_RATE = 24000;

// Modelos que aceptan turn_detection (VAD del servidor).
// Vacío a propósito: NINGUNO de los dos modelos vigentes de este proyecto lo soporta.
// Verificado contra la API real: tanto gpt-realtime-whisper como gpt-live-transcribe responden
// "Turn detection is not supported for this transcription model" y fuerzan turn_detection: null,
// así que el corte de turno se hace con commitTurn() desde el cliente.
// Si en el futuro se reconsidera gpt-transcribe (u otro modelo que sí lo soporte), se agrega acá.
const MODELS_WITH_SERVER_VAD = new Set([]);

// Modelos que usan `languages` (array) en vez de `language` (string).
// gpt-live-transcribe rechaza `language`; gpt-realtime-whisper rechaza `languages`,
// por eso no está acá y cae en el `language` singular por defecto.
const MODELS_WITH_LANGUAGES_ARRAY = new Set(["gpt-live-transcribe"]);

export function modelSupportsServerVad(model) {
  return MODELS_WITH_SERVER_VAD.has(model);
}

// Devuelve el bloque audio.input.transcription con el campo de idioma que acepta cada modelo.
function buildTranscriptionConfig(model, language, { prompt, delay, keywords } = {}) {
  const cfg = { model };
  if (language) {
    if (MODELS_WITH_LANGUAGES_ARRAY.has(model)) cfg.languages = [language];
    else cfg.language = language;
  }
  if (prompt) cfg.prompt = prompt;
  if (keywords?.length) cfg.keywords = keywords;
  // `delay` (minimal|low|medium|high|xhigh) solo lo acepta gpt-live-transcribe.
  if (delay && model === "gpt-live-transcribe") cfg.delay = delay;
  return cfg;
}

/**
 * Abre una sesión de transcripción en vivo y deja el socket listo para recibir audio.
 *
 * @param {object}   opts
 * @param {string}   [opts.model]       Modelo de transcripción (default: OPENAI_REALTIME_TRANSCRIBE_MODEL).
 * @param {string}   [opts.language]    Idioma ISO-639-1 esperado (default: "es").
 * @param {object}   [opts.vad]         Ajustes de server_vad (threshold, prefix_padding_ms, silence_duration_ms).
 * @param {string}   [opts.prompt]      Contexto libre para guiar la transcripción.
 * @param {string[]} [opts.keywords]    Términos literales esperados (nombres propios, siglas).
 * @param {string}   [opts.delay]       Trade-off latencia/precisión, solo gpt-live-transcribe.
 * @param {Function} [opts.onTranscript] (texto, evento) cuando se cierra un turno.
 * @param {Function} [opts.onDelta]      (delta, evento) por cada fragmento parcial.
 * @param {Function} [opts.onSpeechStart]/[opts.onSpeechStop] eventos de VAD del servidor.
 * @param {Function} [opts.onEvent]      Todos los eventos crudos del servidor.
 * @param {Function} [opts.onError]      Errores de la API o del socket.
 * @returns {Promise<RealtimeTranscriptionSession>}
 */
export function openRealtimeTranscriptionSession({
  model = OPENAI_REALTIME_TRANSCRIBE_MODEL,
  language = "es",
  vad = {},
  prompt,
  keywords,
  delay,
  onTranscript,
  onDelta,
  onSpeechStart,
  onSpeechStop,
  onEvent,
  onError,
  timeoutMs = 15000,
} = {}) {
  return new Promise((resolve, reject) => {
    const usesServerVad = modelSupportsServerVad(model);

    const ws = new WebSocket(REALTIME_URL, {
      headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    });

    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { ws.close(); } catch {}
      reject(new Error(`Timeout (${timeoutMs}ms) configurando la sesión de transcripción`));
    }, timeoutMs);

    const fail = (err) => {
      if (settled) { onError?.(err); return; }
      settled = true;
      clearTimeout(timer);
      try { ws.close(); } catch {}
      reject(err);
    };

    ws.on("open", () => {
      const input = {
        format: { type: "audio/pcm", rate: PCM_SAMPLE_RATE },
        transcription: buildTranscriptionConfig(model, language, { prompt, delay, keywords }),
        // Si el modelo no soporta VAD hay que mandar null explícito: la sesión arranca
        // con server_vad por defecto y dejarlo puesto hace que la API rechace el update.
        turn_detection: usesServerVad
          ? {
              type: "server_vad",
              threshold: vad.threshold ?? 0.5,
              prefix_padding_ms: vad.prefix_padding_ms ?? 300,
              silence_duration_ms: vad.silence_duration_ms ?? 500,
            }
          : null,
      };

      ws.send(JSON.stringify({
        type: "session.update",
        session: { type: "transcription", audio: { input } },
      }));
    });

    ws.on("message", (raw) => {
      let event;
      try {
        event = JSON.parse(raw.toString());
      } catch (err) {
        onError?.(new Error(`Evento no parseable: ${err.message}`));
        return;
      }

      onEvent?.(event);

      switch (event.type) {
        case "session.updated":
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            resolve(makeSession(ws, event.session, usesServerVad));
          }
          break;

        case "conversation.item.input_audio_transcription.delta":
          onDelta?.(event.delta || "", event);
          break;

        case "conversation.item.input_audio_transcription.completed":
          onTranscript?.(event.transcript || "", event);
          break;

        case "conversation.item.input_audio_transcription.failed":
          onError?.(new Error(`Transcripción fallida: ${event.error?.message || "desconocido"}`));
          break;

        case "input_audio_buffer.speech_started":
          onSpeechStart?.(event);
          break;

        case "input_audio_buffer.speech_stopped":
          onSpeechStop?.(event);
          break;

        case "error":
          fail(new Error(
            `Realtime API: ${event.error?.message || "error desconocido"} ` +
            `(code=${event.error?.code} param=${event.error?.param})`
          ));
          break;
      }
    });

    ws.on("error", (err) => fail(err));

    ws.on("close", (code, reason) => {
      if (!settled) {
        fail(new Error(`Socket cerrado antes de configurar la sesión (${code} ${reason?.toString() || ""})`));
      }
    });
  });
}

function makeSession(ws, session, usesServerVad) {
  return {
    id: session?.id,
    /** true si el corte de turno lo hace el servidor (VAD); si es false hay que llamar commitTurn(). */
    usesServerVad,
    /** Config efectiva que devolvió la API, útil para verificar qué aceptó realmente. */
    session,
    socket: ws,

    /** Envía un chunk de PCM 16-bit mono @24kHz (Buffer o Uint8Array). */
    sendAudioChunk(chunk) {
      if (ws.readyState !== WebSocket.OPEN) throw new Error("La sesión de transcripción no está abierta");
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      ws.send(JSON.stringify({
        type: "input_audio_buffer.append",
        audio: buf.toString("base64"),
      }));
    },

    /**
     * Cierra el turno a mano. Solo hace falta cuando usesServerVad es false
     * (gpt-live-transcribe / gpt-realtime-whisper no soportan VAD del servidor).
     */
    commitTurn() {
      if (ws.readyState !== WebSocket.OPEN) throw new Error("La sesión de transcripción no está abierta");
      ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
    },

    close() {
      try { ws.close(1000, "OK"); } catch {}
    },
  };
}
