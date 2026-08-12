// index.js
import { exec } from "child_process";
import cors from "cors";
import dotenv from "dotenv";
import voice from "elevenlabs-node";
import express from "express";
import { promises as fs } from "fs";
import OpenAI from "openai";
import { toFile } from "openai/uploads";
import path from "path";
import { File } from "node:buffer";
if (!globalThis.File) globalThis.File = File;

dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || "-" });
const elevenLabsApiKey = process.env.ELEVEN_LABS_API_KEY;
const voiceID = "86V9x9hrQds83qf7zaGn";

const app = express();
app.use(cors());
app.use(express.json({ limit: "100mb" })); // ok aunque /reset no mande body
const port = 3000;

const MODELS_BASE_URL = process.env.MIA_MODELS_URL || "http://localhost:8001";

const HIST_DIR = path.join("historial");
const HIST_PATH = path.join(HIST_DIR, "historial.json");
const SUMMARY_PATH = path.join(HIST_DIR, "historial_resumen.json");

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

// ====== Shell ======
const execCommand = (command) => new Promise((resolve, reject) => { exec(command, (error, stdout, stderr) => { if (error) { console.error("Command error:", stderr || error.message); return reject(error); } resolve(stdout); }); });


async function ensureDirs() {
  await fs.mkdir(HIST_DIR, { recursive: true });
  await fs.mkdir("audios", { recursive: true });
}
async function readJsonSafe(filePath, fallback = {}) {
  try { return JSON.parse(await fs.readFile(filePath, "utf8")); }
  catch { return fallback; }
}
async function writeJson(filePath, obj) {
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2));
}
async function clearAudios() {
  await fs.rm("audios", { recursive: true, force: true });
  await fs.mkdir("audios", { recursive: true });
}
async function clearHistorial() {
  await fs.mkdir(HIST_DIR, { recursive: true });
  await writeJson(HIST_PATH, {});
  await writeJson(SUMMARY_PATH, {});
}
async function resetSession(reason = "manual") {
  log(`Reset de sesión (${reason}) → limpiando audios e historial`);
  await clearAudios();
  await clearHistorial();
}
// ====== Utilidades de archivo para audios/lipsync ====== 
const readJsonTranscript = async (file) => { const data = await fs.readFile(file, "utf8"); return JSON.parse(data); }; 
const audioFileToBase64 = async (file) => { const data = await fs.readFile(file); return data.toString("base64"); };

const lipSyncMessage = async (messageIndex) => {
  const time = Date.now();
  await execCommand(`ffmpeg -y -i audios/message_${messageIndex}.mp3 audios/message_${messageIndex}.wav`);
  await execCommand(`rhubarb -f json -o audios/message_${messageIndex}.json audios/message_${messageIndex}.wav -r phonetic`);
  console.log(`Lip sync done in ${Date.now() - time}ms`);
};

async function transcribeBufferWithWhisper(buffer, filename, mime) {
  const file = await toFile(buffer, filename, { type: mime });
  const resp = await openai.audio.transcriptions.create({
    model: "whisper-1",
    file,
    language: "es",
    response_format: "json",
  });
  return resp.text || "";
}

function parseDataUrl(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.slice(5, header.indexOf(";")) || "audio/webm";
  const ext = mime.split("/")[1] || "webm";
  const buffer = Buffer.from(base64, "base64");
  return { buffer, mime, ext };
}

async function callLocalSentiment(text) {
  const r = await fetch(`${MODELS_BASE_URL}/sentiment`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`sentiment ${r.status}`);
  const j = await r.json();
  return (j.sentimiento || "").toLowerCase();
}

async function callLocalMiaPredict(text, sentimiento) {
  const r = await fetch(`${MODELS_BASE_URL}/mia_predict`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, sentimiento }),
  });
  if (!r.ok) throw new Error(`mia_predict ${r.status}`);
  const j = await r.json();
  return (j.mia_emocion || "").toLowerCase();
}

function rotatingTalkingAnimation(conversationIndex) {
  const mod = conversationIndex % 3;
  if (mod === 0) return "Talking_0";
  if (mod === 1) return "Talking_1";
  return "Talking_2";
}
function mapEmotionToVisuals(mia_emocion, conversationIndex) {
  const e = (mia_emocion || "default").toLowerCase();
  let facialExpression = "default";
  if (e === "alegría" || e === "amor") facialExpression = "smile";
  else if (e === "tristeza") facialExpression = "sad";
  else if (e === "ira") facialExpression = "angry";
  else if (e === "miedo" || e === "sorpresa") facialExpression = "surprised";
  return { facialExpression, animation: rotatingTalkingAnimation(conversationIndex) };
}

function getVoiceSettingsForEmotion(mia_emocion) {
  if (mia_emocion === "alegría") {
    return { stability: 0.3, similarityBoost: 0.75 };
  }
  if (mia_emocion === "amor") {
    return { stability: 0.45, similarityBoost: 0.8 };
  }
  return { stability: 0.5, similarityBoost: 0.75 };
}

const MAX_USER_MSG_CHARS = 500;

function sanitizeSummary(text) {
  if (!text) return text;
  return text
    .replace(/^\s*\[.*?\]\s*/gm, "")   // saca cualquier [encabezado] al inicio de línea
    .replace(/^\s*[-•]\s*/gm, "")       // saca viñetas sueltas
    .replace(/\n{2,}/g, " ")             // colapsa saltos de línea múltiples
    .replace(/\n/g, " ")                  // colapsa saltos de línea simples
    .trim();
}

async function updateRollingSummary(latestTurn) {
  const summaryCurrent = await readJsonSafe(SUMMARY_PATH, {});
  const resumenPrevio = summaryCurrent?.resumen || "";

  const system = `
Eres un sistema que mantiene un resumen narrativo compacto de una conversación de acompañamiento emocional entre un usuario y MIA.

Reglas estrictas que debés seguir siempre:
- Reescribí el resumen completo desde cero, integrando lo nuevo. NUNCA agregues el turno nuevo como una línea extra al final del resumen anterior — sintetizá todo junto en un texto nuevo y más corto si hace falta.
- Formato: un único párrafo en prosa corrida. Sin viñetas, sin guiones, sin títulos, sin corchetes, sin las etiquetas "Usuario:" o "MIA:".
- Extensión máxima: 3 a 4 oraciones en total, sin importar cuántos turnos lleve la conversación. Si ya es largo, comprimí más en vez de seguir agregando.
- Parafraseá el contenido y el estado emocional; no cites frases textuales de lo que dijeron.
- Tu respuesta completa ES el resumen y nada más. No incluyas encabezados ni texto explicativo.
- Escribí en español, sin emojis.

Ejemplo de formato correcto:
"El usuario suele saludar con ánimo alegre y valora que MIA le pregunte por su día. En la conversación más reciente contó que perdió a su gato y se sintió triste; MIA lo acompañó validando su tristeza y ofreciéndose a escuchar más."

Ejemplo de formato INCORRECTO (no hagas esto):
"[Resumen acumulado actualizado]
- Usuario: saluda alegre.
- MIA: pregunta cómo está."
`.trim();

  const user = `
Resumen actual: ${resumenPrevio || "(vacío, es el primer turno)"}

Nuevo turno — el usuario dijo: "${latestTurn.user_responde}", y MIA respondió: "${latestTurn.mia_text}".

Reescribí el resumen completo actualizado siguiendo las reglas del system prompt.
`.trim();

  const resp = await openai.responses.create({
    model: "gpt-5-nano",
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  });

  const resumen = sanitizeSummary((resp.output_text || resumenPrevio || "").trim());
  await writeJson(SUMMARY_PATH, {
    resumen,
    updatedAt: new Date().toISOString(),
  });
}

async function condenseUserMessageIfNeeded(transcript) {
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
    model: "gpt-5-nano",
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  });

  const condensado = (resp.output_text || "").trim();
  return condensado || transcript; // fallback al original si la API devuelve vacío
}

async function generateMiaReply({ transcript, sentimiento, mia_emocion }) {
  const summary = await readJsonSafe(SUMMARY_PATH, {});
  const resumen = summary?.resumen || "";

  const system = `
Eres "MIA", un agente de IA empática y de acompañamiento emocional.
Respondes SIEMPRE en español, con calidez y claridad, sin emojis.
`.trim();

  const user = `
[Resumen de la conversación hasta ahora]
${resumen || "(sin resumen aún, es el primer turno)"}

[Turno actual]
${transcript}

[Metadatos]
- Sentimiento del usuario: ${sentimiento}
- Emoción de respuesta: ${mia_emocion}

[Instrucciones]
- Breve (2–3 frases), valida y acompaña. Sin consejos clínicos.
`.trim();

  const resp = await openai.responses.create({
    model: "gpt-5-nano",
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  });

  return (resp.output_text || "").trim();
}

// ================== ENDPOINT: RESET MANUAL ==================
app.post("/reset", async (req, res) => {
  try {
    await ensureDirs();
    log(`↩️  RESET recibido. query.ts=${req.query?.ts || ""} body=${JSON.stringify(req.body || {})}`);
    await resetSession(req.body?.reason || "manual");
    return res.json({ ok: true, reset: true });
  } catch (e) {
    console.error("Error en /reset:", e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

// ================== ENDPOINT PRINCIPAL ==================
app.post("/chat", async (req, res) => {
  try {
    await ensureDirs();
    const userMessage = req.body.message;

    if (typeof userMessage === "string" && userMessage.startsWith("data:audio")) {
      log("🎙️ Audio recibido, transcribiendo...");
      const { buffer, mime, ext } = parseDataUrl(userMessage);
      let transcript = await transcribeBufferWithWhisper(buffer, `audio.${ext}`, mime);

      transcript = await condenseUserMessageIfNeeded(transcript);

      let sentimiento = "neutral";
      let mia_emocion = "default";
      try { sentimiento = await callLocalSentiment(transcript); } catch (e) { console.warn("sentiment:", e.message); }
      try { mia_emocion = await callLocalMiaPredict(transcript, sentimiento); } catch (e) { console.warn("mia_predict:", e.message); }

      const historialActual = await readJsonSafe(HIST_PATH, {});
      const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
      const nextKey = `conversacion_${nextIndex}`;

      const mia_text = await generateMiaReply({ transcript, sentimiento, mia_emocion });
      const visuals = mapEmotionToVisuals(mia_emocion, nextIndex - 1);

      // Guardamos el texto ANTES de tocar audio: si ElevenLabs falla, el turno no se pierde.
      historialActual[nextKey] = { user_responde: transcript, sentimiento, mia_emocion, mia_text };
      await writeJson(HIST_PATH, historialActual);
      await updateRollingSummary(historialActual[nextKey]);

      log(`💾 Turno guardado como ${nextKey}, generando audio...`);

      let audio;
      let lipsync;
      try {
        const idx = 0;
        const fileName = `audios/message_${idx}.mp3`;
        const { stability, similarityBoost } = getVoiceSettingsForEmotion(mia_emocion);

        await voice.textToSpeech(
          elevenLabsApiKey,
          voiceID,
          fileName,
          mia_text,
          stability,
          similarityBoost,
          "eleven_multilingual_v2"
        );

        const stats = await fs.stat(fileName).catch(() => null);
        if (!stats || stats.size === 0) {
          throw new Error(`ElevenLabs no generó audio válido en ${fileName} (archivo inexistente o vacío). Revisa la API key, el voiceID o la cuota de ElevenLabs.`);
        }

        await lipSyncMessage(idx);
        audio = await audioFileToBase64(fileName);
        lipsync = await readJsonTranscript(`audios/message_${idx}.json`);

        log(`✅ Audio flujo completo generado para ${nextKey}`);
      } catch (audioErr) {
        console.error("Error generando audio/lipsync:", audioErr);
      }

      return res.json({
        ok: true,
        transcript,
        sentimiento,
        mia_emocion,
        messages: [
          {
            text: mia_text,
            audio,
            lipsync,
            facialExpression: visuals.facialExpression,
            animation: visuals.animation,
          },
        ],
      });
    }

    // Texto sin audio → saludo on-the-fly (sin intro_*.wav)
    if (!userMessage) {
      const mia_text = "Estoy aquí. ¿Qué te gustaría contarme?";
      const historialActual = await readJsonSafe(HIST_PATH, {});
      const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
      const visuals = mapEmotionToVisuals("default", nextIndex - 1);

      try {
        const idx = 0;
        const fileName = `audios/message_${idx}.mp3`;
        await voice.textToSpeech(
          elevenLabsApiKey,
          voiceID,
          fileName,
          mia_text,
          0.5,
          0.75,
          "eleven_multilingual_v2"
        );
        await lipSyncMessage(idx);
        const audio = await audioFileToBase64(fileName);
        const lipsync = await readJsonTranscript(`audios/message_${idx}.json`);
        return res.json({ ok: true, messages: [{ text: mia_text, audio, lipsync, facialExpression: visuals.facialExpression, animation: visuals.animation }] });
      } catch {
        return res.json({ ok: true, messages: [{ text: mia_text, facialExpression: visuals.facialExpression, animation: visuals.animation }] });
      }
    }

    // (Tu rama vieja de chat JSON con 3.5 se mantiene por compat)
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo-1106",
      max_tokens: 1000,
      temperature: 0.6,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: `
          You are a virtual girlfriend.
          You will always reply with a JSON array of messages. With a maximum of 3 messages.
          Each message has a text, facialExpression, and animation property.
          The animations allowed are only Talking_0, Talking_1, Talking_2.
        `},
        { role: "user", content: userMessage },
      ],
    });

    let messages = JSON.parse(completion.choices[0].message.content);
    if (messages.messages) messages = messages.messages;

    await fs.mkdir("audios", { recursive: true });
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      const fileName = `audios/message_${i}.mp3`;
      await voice.textToSpeech(
        elevenLabsApiKey,
        voiceID,
        fileName,
        m.text,
        0.5,
        0.75,
        "eleven_multilingual_v2"
      );
      await lipSyncMessage(i);
      m.audio = await audioFileToBase64(fileName);
      m.lipsync = await readJsonTranscript(`audios/message_${i}.json`);
    }

    res.send({ messages });
  } catch (err) {
    console.error("Error en /chat:", err);
    res.status(500).json({ error: err.message, messages: [] });
  }
});

app.listen(port, () => {
  console.log(`Virtual Girlfriend listening on port ${port}`);
});
