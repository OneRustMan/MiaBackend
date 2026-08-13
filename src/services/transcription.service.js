// src/services/transcription.service.js
import { toFile } from "openai/uploads";
import { openai } from "../clients/openaiClient.js";
import { MAX_USER_MSG_CHARS, OPENAI_TEXT_MODEL, OPENAI_TRANSCRIBE_MODEL } from "../config/env.js";

export async function transcribeBufferWithWhisper(buffer, filename, mime, signal) {
  const file = await toFile(buffer, filename, { type: mime });
  const resp = await openai.audio.transcriptions.create({
    model: OPENAI_TRANSCRIBE_MODEL,
    file,
    language: "es",
    response_format: "json",
  }, { signal });
  return resp.text || "";
}

export function parseDataUrl(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.slice(5, header.indexOf(";")) || "audio/webm";
  const ext = mime.split("/")[1] || "webm";
  const buffer = Buffer.from(base64, "base64");
  return { buffer, mime, ext };
}

export async function condenseUserMessageIfNeeded(transcript, signal) {
  if (!transcript || transcript.length <= MAX_USER_MSG_CHARS) return transcript;

  const system = `
Eres un sistema que condensa mensajes largos de un usuario, preservando el sentido, el tono emocional y los datos concretos (nombres, lugares, hechos) que mencione.
Devuelve SOLO el mensaje condensado, en español, sin comillas ni explicaciones adicionales.
`.trim();

  const user = `
Condensa el siguiente mensaje del usuario a un máximo de 2-3 frases, sin perder el sentido:

"${transcript}"
`.trim();

  const resp = await openai.responses.create({
    model: OPENAI_TEXT_MODEL,
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  }, { signal });

  const condensado = (resp.output_text || "").trim();
  return condensado || transcript; // fallback al original si la API devuelve vacío
}
