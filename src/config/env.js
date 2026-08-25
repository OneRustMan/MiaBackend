// src/config/env.js
import dotenv from "dotenv";
import path from "path";

dotenv.config();

export const PORT = 3000;

export const MODELS_BASE_URL = process.env.MIA_MODELS_URL || "http://localhost:8001";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "-";
export const ELEVEN_LABS_API_KEY = process.env.ELEVEN_LABS_API_KEY;
export const VOICE_ID = process.env.ELEVEN_LABS_VOICE_ID || "86V9x9hrQds83qf7zaGn";

export const OPENAI_TEXT_MODEL = process.env.OPENAI_TEXT_MODEL || "gpt-5-nano";
export const OPENAI_TRANSCRIBE_MODEL = process.env.OPENAI_TRANSCRIBE_MODEL || "whisper-1";
export const ELEVEN_LABS_MODEL_ID = process.env.ELEVEN_LABS_MODEL_ID || "eleven_multilingual_v2";

// Rutas relativas al cwd desde donde se arranca `node index.js` (no a __dirname).
export const HIST_DIR = path.join("historial");
export const HIST_PATH = path.join(HIST_DIR, "historial.json");
export const SUMMARY_PATH = path.join(HIST_DIR, "historial_resumen.json");

export const MAX_USER_MSG_CHARS = 500;

// Modelo de transcripción en vivo (Realtime API). Intercambiable por .env para poder
// comparar modelos sin tocar código, igual que OPENAI_TEXT_MODEL / OPENAI_TRANSCRIBE_MODEL.
// Ojo: no todos los modelos aceptan las mismas opciones (ver realtimeTranscription.service.js).
export const OPENAI_REALTIME_TRANSCRIBE_MODEL =
  process.env.OPENAI_REALTIME_TRANSCRIBE_MODEL || "gpt-realtime-whisper";
