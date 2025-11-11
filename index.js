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
const voiceID = "5vkxOzoz40FrElmLP4P7";

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

const MAX_CONTEXT_CHARS = 10000;
const RECENT_TURNS = 6;

function formatRecentTurns(historialObj) {
  const keys = Object.keys(historialObj)
    .filter(k => k.startsWith("conversacion_"))
    .sort((a, b) => (parseInt(a.split("_")[1]||"0",10) - parseInt(b.split("_")[1]||"0",10)));
  const slice = keys.slice(-RECENT_TURNS);
  let acc = "";
  for (const key of slice) {
    const c = historialObj[key];
    acc += `\n[${key}]\nUsuario: ${c?.user_responde ?? ""}\nMIA: ${c?.mia_text ?? "(sin respuesta)"}\n`;
  }
  return acc.trim();
}

async function maybeUpdateSummary() {
  const historial = await readJsonSafe(HIST_PATH, {});
  const serialized = JSON.stringify(historial);
  if (serialized.length <= MAX_CONTEXT_CHARS) return;

  const summaryCurrent = await readJsonSafe(SUMMARY_PATH, {});
  const resumenPrevio = summaryCurrent?.resumen || "";

  const textToSummarize = `
[Resumen previo]
${resumenPrevio}

[Historial completo (compactar lo relevante)]
${Object.keys(historial).map(k => {
  const c = historial[k];
  return `${k}:
Usuario: ${c?.user_responde || ""}
MIA: ${c?.mia_text || ""}
Sentimiento: ${c?.sentimiento || ""}
MIA_emoción: ${c?.mia_emocion || ""}`;
}).join("\n\n")}
`.trim();

  const system = `
Eres un sistema que resume conversaciones de acompañamiento emocional.
Devuelve un resumen breve, fiel y útil para continuar la conversación en turnos futuros.
Escribe en español. Sin emojis.
`.trim();

  const user = `
Resume de forma compacta el siguiente material. 
${textToSummarize}
`.trim();

  const resp = await openai.responses.create({
    model: "gpt-5-nano",
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  });

  const resumen = (resp.output_text || "").trim();
  await writeJson(SUMMARY_PATH, {
    resumen,
    updatedAt: new Date().toISOString(),
    total_conversaciones: Object.keys(historial).length,
  });
}

async function loadContextForLLM() {
  const historial = await readJsonSafe(HIST_PATH, {});
  const summary = await readJsonSafe(SUMMARY_PATH, {});
  const recent = formatRecentTurns(historial);
  return {
    resumen: summary?.resumen || "",
    recientes: recent,
    total_conversaciones: Object.keys(historial).length,
  };
}

async function generateMiaReply({ transcript, sentimiento, mia_emocion }) {
  const { resumen, recientes, total_conversaciones } = await loadContextForLLM();

  const system = `
Eres "MIA", un agente de IA empática y de acompañamiento emocional.
Respondes SIEMPRE en español, con calidez y claridad, sin emojis.
`.trim();

  const user = `
[Memoria Resumida]
${resumen || "(sin resumen aún)"}

[Últimas interacciones relevantes]
${recientes || "(no hay historial reciente)"}

[Turno actual]
${transcript}

[Metadatos]
- Sentimiento del usuario: ${sentimiento}
- Emoción de respuesta: ${mia_emocion}
- Total conversaciones: ${total_conversaciones}

[Instrucciones]
- Breve (2–4 frases), valida y acompaña. Sin consejos clínicos.
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
      const transcript = await transcribeBufferWithWhisper(buffer, `audio.${ext}`, mime);

      let sentimiento = "neutral";
      let mia_emocion = "default";
      try { sentimiento = await callLocalSentiment(transcript); } catch (e) { console.warn("sentiment:", e.message); }
      try { mia_emocion = await callLocalMiaPredict(transcript, sentimiento); } catch (e) { console.warn("mia_predict:", e.message); }

      const historialActual = await readJsonSafe(HIST_PATH, {});
      const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
      const nextKey = `conversacion_${nextIndex}`;

      const mia_text = await generateMiaReply({ transcript, sentimiento, mia_emocion });
      const visuals = mapEmotionToVisuals(mia_emocion, nextIndex - 1);

      const idx = 0;
      const fileName = `audios/message_${idx}.mp3`;
      await voice.textToSpeech(elevenLabsApiKey, voiceID, fileName, mia_text);
      await lipSyncMessage(idx);
      const audio = await audioFileToBase64(fileName);
      const lipsync = await readJsonTranscript(`audios/message_${idx}.json`);

      historialActual[nextKey] = { user_responde: transcript, sentimiento, mia_emocion, mia_text };
      await writeJson(HIST_PATH, historialActual);
      await maybeUpdateSummary();

      log(`✅ Audio flujo completo generado y guardado como ${nextKey}`);

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
        await voice.textToSpeech(elevenLabsApiKey, voiceID, fileName, mia_text);
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
      await voice.textToSpeech(elevenLabsApiKey, voiceID, fileName, m.text);
      await lipSyncMessage(i);
      m.audio = await audioFileToBase64(fileName);
      m.lipsync = await readJsonTranscript(`audios/message_${i}.json`);
    }

    res.send({ messages });
  } catch (err) {
    console.error("Error en /chat:", err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(port, () => {
  console.log(`Virtual Girlfriend listening on port ${port}`);
});
