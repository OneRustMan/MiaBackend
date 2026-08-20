// src/services/audio.service.js
import { promises as fs } from "fs";

import { ELEVEN_LABS_API_KEY, ELEVEN_LABS_MODEL_ID, VOICE_ID } from "../config/env.js";
import { execCommand } from "../utils/shell.js";

export async function clearAudios() {
  await fs.rm("audios", { recursive: true, force: true });
  await fs.mkdir("audios", { recursive: true });
}

// ====== Utilidades de archivo para audios/lipsync ======
export const readJsonTranscript = async (file) => { const data = await fs.readFile(file, "utf8"); return JSON.parse(data); };
export const audioFileToBase64 = async (file) => { const data = await fs.readFile(file); return data.toString("base64"); };

export const lipSyncMessage = async (messageIndex) => {
  const time = Date.now();
  await execCommand(`ffmpeg -y -i audios/message_${messageIndex}.mp3 audios/message_${messageIndex}.wav`);
  await execCommand(`rhubarb -f json -o audios/message_${messageIndex}.json audios/message_${messageIndex}.wav -r phonetic`);
  console.log(`Lip sync done in ${Date.now() - time}ms`);
};

// ====== TTS con timings en la misma respuesta (sin disco, sin ffmpeg/rhubarb) ======

// Mapeo heurístico de carácter a viseme estilo Rhubarb (A-H, X).
// Es una aproximación por letra, no un análisis fonético real como el
// que hacía rhubarb — más rápido, algo menos preciso. Ajustable a
// futuro si al probarlo visualmente se ve mal en algún sonido puntual.
function charToViseme(char) {
  const c = char.toLowerCase();
  if (/[\s.,;:!?¡¿"'()]/.test(c)) return "X";
  if ("aá".includes(c)) return "D";
  if ("eé".includes(c)) return "C";
  if ("ií".includes(c)) return "B";
  if ("oó".includes(c)) return "E";
  if ("uú".includes(c)) return "F";
  if ("mbp".includes(c)) return "A";
  if ("fv".includes(c)) return "G";
  if (c === "l") return "H";
  return "B"; // default para el resto de consonantes
}

// Agrupa caracteres consecutivos con el mismo viseme en un solo cue,
// para que la animación no sea más entrecortada de lo necesario.
function buildMouthCues(characters, startTimes, endTimes) {
  const cues = [];
  for (let i = 0; i < characters.length; i++) {
    const value = charToViseme(characters[i]);
    const last = cues[cues.length - 1];
    if (last && last.value === value) {
      last.end = endTimes[i];
    } else {
      cues.push({ start: startTimes[i], end: endTimes[i], value });
    }
  }
  return cues;
}

export async function generateSpeechWithTimestamps(text, stability, similarityBoost, signal) {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/with-timestamps`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: ELEVEN_LABS_MODEL_ID,
        voice_settings: { stability, similarity_boost: similarityBoost },
      }),
      signal,
    }
  );

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`ElevenLabs with-timestamps ${response.status}: ${detail}`);
  }

  const data = await response.json();
  if (!data.audio_base64) {
    throw new Error("ElevenLabs with-timestamps no devolvió audio_base64.");
  }

  const alignment = data.alignment || data.normalized_alignment;
  const mouthCues = alignment
    ? buildMouthCues(alignment.characters, alignment.character_start_times_seconds, alignment.character_end_times_seconds)
    : [];

  return {
    audio: data.audio_base64,
    lipsync: { metadata: { soundFile: "elevenlabs-stream", duration: mouthCues.at(-1)?.end || 0 }, mouthCues },
  };
}
