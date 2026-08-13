// src/services/greeting.service.js
import path from "path";
import { audioFileToBase64, readJsonTranscript } from "./audio.service.js";

export const GREETING_TEXT = "Estoy aquí. ¿Qué te gustaría contarme?";

// Contenido fijo del proyecto: assets/ NO se limpia nunca en resetSession().
const GREETING_AUDIO_PATH = path.join("assets", "greeting", "greeting.mp3");
const GREETING_LIPSYNC_PATH = path.join("assets", "greeting", "greeting.json");

let cachedGreetingAudio = null;

export async function getGreetingAudio() {
  if (!cachedGreetingAudio) {
    const audio = await audioFileToBase64(GREETING_AUDIO_PATH);
    const lipsync = await readJsonTranscript(GREETING_LIPSYNC_PATH);
    cachedGreetingAudio = { audio, lipsync };
  }
  return cachedGreetingAudio;
}
