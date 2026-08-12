// src/services/session.service.js
import { log } from "../utils/logger.js";
import { clearAudios } from "./audio.service.js";
import { clearHistorial } from "./historial.service.js";

// Cancelación de sesión: cada /reset aborta el trabajo en vuelo y sube la generación.
// Estas dos variables viven SOLO acá; el resto del código las lee vía getSessionSnapshot().
let resetGeneration = 0;
let sessionController = new AbortController();

export function getSessionSnapshot() {
  return { generation: resetGeneration, signal: sessionController.signal };
}

export async function resetSession(reason = "manual") {
  log(`Reset de sesión (${reason}) → limpiando audios e historial`);
  sessionController.abort();
  sessionController = new AbortController();
  resetGeneration++;
  await clearAudios();
  await clearHistorial();
}
