// src/config/env.js
import dotenv from "dotenv";
import path from "path";

dotenv.config();

export const PORT = 3000;

export const MODELS_BASE_URL = process.env.MIA_MODELS_URL || "http://localhost:8001";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "-";
export const ELEVEN_LABS_API_KEY = process.env.ELEVEN_LABS_API_KEY;
export const VOICE_ID = "86V9x9hrQds83qf7zaGn";

// Rutas relativas al cwd desde donde se arranca `node index.js` (no a __dirname).
export const HIST_DIR = path.join("historial");
export const HIST_PATH = path.join(HIST_DIR, "historial.json");
export const SUMMARY_PATH = path.join(HIST_DIR, "historial_resumen.json");

export const MAX_USER_MSG_CHARS = 500;
