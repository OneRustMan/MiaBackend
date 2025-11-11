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

// ================== OpenAI (Responses API) ==================
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "-",
});

// ================== ElevenLabs ==================
const elevenLabsApiKey = process.env.ELEVEN_LABS_API_KEY;
const voiceID = "5vkxOzoz40FrElmLP4P7";

// ================== Express ==================
const app = express();
app.use(express.json({ limit: "100mb" }));
app.use(cors());
const port = 3000;

// ====== Microservicio de modelos (Python FastAPI) ======
const MODELS_BASE_URL = process.env.MIA_MODELS_URL || "http://localhost:8001";

// ====== Paths de historial/resumen ======
const HIST_DIR = path.join("historial");
const HIST_PATH = path.join(HIST_DIR, "historial.json");
const SUMMARY_PATH = path.join(HIST_DIR, "historial_resumen.json");

// ====== Inactividad (5 min) ======
const INACTIVITY_MS = 5 * 60 * 1000;
let lastActivity = Date.now();
let sessionExpired = false;

// Marca actividad en cada request
app.use((req, _res, next) => {
  // Si ya expiró, no toques lastActivity aquí; se maneja en /chat o /reset
  if (!sessionExpired) lastActivity = Date.now();
  next();
});

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

// ====== Helpers FS ======
async function ensureDirs() {
  await fs.mkdir(HIST_DIR, { recursive: true });
  await fs.mkdir("audios", { recursive: true });
}

async function readJsonSafe(filePath, fallback = {}) {
  try {
    const data = await fs.readFile(filePath, "utf8");
    return JSON.parse(data);
  } catch {
    return fallback;
  }
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
  await writeJson(HIST_PATH, {}); // vacío
  await writeJson(SUMMARY_PATH, {}); // vacío
}

async function resetSession(reason = "manual") {
  log(`Reset de sesión (${reason}) → limpiando audios e historial`);
  await clearAudios();
  await clearHistorial();
  sessionExpired = false;
  lastActivity = Date.now();
}

// Watchdog de inactividad (corre cada 30s)
setInterval(async () => {
  const idle = Date.now() - lastActivity;
  if (!sessionExpired && idle >= INACTIVITY_MS) {
    sessionExpired = true;
    await resetSession("inactivity");
  }
}, 30 * 1000);

// ====== Shell ======
const execCommand = (command) =>
  new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error("Command error:", stderr || error.message);
        return reject(error);
      }
      resolve(stdout);
    });
  });

// ====== Rhubarb (binario global) ======
const lipSyncMessage = async (messageIndex) => {
  const time = Date.now();
  await execCommand(
    `ffmpeg -y -i audios/message_${messageIndex}.mp3 audios/message_${messageIndex}.wav`
  );
  await execCommand(
    `rhubarb -f json -o audios/message_${messageIndex}.json audios/message_${messageIndex}.wav -r phonetic`
  );
  console.log(`Lip sync done in ${Date.now() - time}ms`);
};

// ====== Transcripción (Whisper) ======
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

// ====== DataURL -> buffer ======
function parseDataUrl(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.slice(5, header.indexOf(";")) || "audio/webm";
  const ext = mime.split("/")[1] || "webm";
  const buffer = Buffer.from(base64, "base64");
  return { buffer, mime, ext };
}

// ====== Microservicio local ======
async function callLocalSentiment(text) {
  const r = await fetch(`${MODELS_BASE_URL}/sentiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`sentiment ${r.status}`);
  const j = await r.json();
  return (j.sentimiento || "").toLowerCase();
}

async function callLocalMiaPredict(text, sentimiento) {
  const r = await fetch(`${MODELS_BASE_URL}/mia_predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, sentimiento }),
  });
  if (!r.ok) throw new Error(`mia_predict ${r.status}`);
  const j = await r.json();
  return (j.mia_emocion || "").toLowerCase();
}

// ====== Visuales: expresiones + SOLO Talking_0..2 ======
function rotatingTalkingAnimation(conversationIndex) {
  const mod = conversationIndex % 3;
  if (mod === 0) return "Talking_0";
  if (mod === 1) return "Talking_1";
  return "Talking_2";
}

function mapEmotionToVisuals(mia_emocion, conversationIndex) {
  const e = (mia_emocion || "default").toLowerCase();
  let facialExpression = "default";
  if (e === "alegría") facialExpression = "smile";
  else if (e === "amor") facialExpression = "smile";
  else if (e === "tristeza") facialExpression = "sad";
  else if (e === "ira") facialExpression = "angry";
  else if (e === "miedo") facialExpression = "surprised";
  else if (e === "sorpresa") facialExpression = "surprised";

  return { facialExpression, animation: rotatingTalkingAnimation(conversationIndex) };
}

// ====== Contexto al LLM: resumen + últimas K ======
const MAX_CONTEXT_CHARS = 10000;
const RECENT_TURNS = 6;

function formatRecentTurns(historialObj) {
  const keys = Object.keys(historialObj)
    .filter(k => k.startsWith("conversacion_"))
    .sort((a, b) => {
      const na = parseInt(a.split("_")[1] || "0", 10);
      const nb = parseInt(b.split("_")[1] || "0", 10);
      return na - nb;
    });

  const slice = keys.slice(-RECENT_TURNS);
  let acc = "";
  for (const key of slice) {
    const c = historialObj[key];
    const u = c?.user_responde ?? "";
    const m = c?.mia_text ?? "";
    acc += `\n[${key}]\nUsuario: ${u}\nMIA: ${m || "(sin respuesta registrada)"}\n`;
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

[Historial completo (compactar y extraer solo lo relevante para continuidad emocional y hechos clave)]
${Object.keys(historial)
  .map(k => {
    const c = historial[k];
    return `${k}:
Usuario: ${c?.user_responde || ""}
MIA: ${c?.mia_text || ""}
Sentimiento: ${c?.sentimiento || ""}
MIA_emoción: ${c?.mia_emocion || ""}`;
  })
  .join("\n\n")}
`.trim();

  const system = `
Eres un sistema que resume conversaciones de acompañamiento emocional.
Devuelve un resumen breve, fiel y útil para continuar la conversación en turnos futuros.
Escribe en español. Sin emojis. Mantén nombres, metas del usuario, temas sensibles y preferencias que hayan surgido.
`.trim();

  const user = `
Resume de forma compacta el siguiente material. Evita repetir texto literal, extrae hechos, contexto emocional y objetivos.
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

  const newSummary = {
    resumen,
    updatedAt: new Date().toISOString(),
    total_conversaciones: Object.keys(historial).length,
  };
  await writeJson(SUMMARY_PATH, newSummary);
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

// ====== Generación de respuesta MIA ======
async function generateMiaReply({ transcript, sentimiento, mia_emocion }) {
  const { resumen, recientes, total_conversaciones } = await loadContextForLLM();

  const system = `
Eres "MIA", un agente de IA empática y de acompañamiento emocional.
Respondes SIEMPRE en español, con calidez y claridad, sin emojis.
Adapta el tono a la emoción final con la que debes responder (mia_emocion).
Evita consejos clínicos; valida emociones, ofrece presencia y sugerencias prácticas suaves.
`.trim();

  const user = `
[Memoria Resumida]
${resumen || "(sin resumen aún)"}

[Últimas interacciones relevantes]
${recientes || "(no hay historial reciente)"}

[Turno actual (transcripción)]
${transcript}

[Metadatos]
- Sentimiento del usuario: ${sentimiento}
- Emoción con la que debe responder MIA: ${mia_emocion}
- Conteo total de conversaciones: ${total_conversaciones}

[Instrucciones]
- Sin emojis.
- Empática, cercana y breve (2–4 frases).
- Si corresponde, celebra avances (alegría/amor). Si hay malestar, valida y acompaña sin juzgar.
`.trim();

  const resp = await openai.responses.create({
    model: "gpt-5-nano",
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  });

  const text = (resp.output_text || "").trim();
  return text; // sin fallbacks
}

// ====== Utilidades de archivo para audios/lipsync ======
const readJsonTranscript = async (file) => {
  const data = await fs.readFile(file, "utf8");
  return JSON.parse(data);
};

const audioFileToBase64 = async (file) => {
  const data = await fs.readFile(file);
  return data.toString("base64");
};

// ================== ENDPOINT: RESET MANUAL ==================
app.post("/reset", async (_req, res) => {
  try {
    await resetSession("manual");
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

    // Si la sesión está expirada por inactividad, corta aquí y limpia
    if (sessionExpired || (Date.now() - lastActivity) >= INACTIVITY_MS) {
      await resetSession("on-chat-expired");
      return res.json({
        ok: false,
        sessionExpired: true,
        messages: [],
      });
    }

    const userMessage = req.body.message;

    // ======= CASO 1: Audio =========
    if (typeof userMessage === "string" && userMessage.startsWith("data:audio")) {
      log("🎙️ Audio recibido, transcribiendo...");
      const { buffer, mime, ext } = parseDataUrl(userMessage);
      const transcript = await transcribeBufferWithWhisper(buffer, `audio.${ext}`, mime);

      // Analítica local
      let sentimiento = "neutral";
      let mia_emocion = "default";
      try {
        sentimiento = await callLocalSentiment(transcript);
      } catch (e) {
        console.warn("⚠️ sentiment local error:", e.message);
      }
      try {
        mia_emocion = await callLocalMiaPredict(transcript, sentimiento);
      } catch (e) {
        console.warn("⚠️ mia_predict local error:", e.message);
      }

      // Número de conversación siguiente
      const historialActual = await readJsonSafe(HIST_PATH, {});
      const nextIndex = Object.keys(historialActual).filter(k => k.startsWith("conversacion_")).length + 1;
      const nextKey = `conversacion_${nextIndex}`;

      // Generación de texto empático
      const mia_text = await generateMiaReply({ transcript, sentimiento, mia_emocion });

      // Visuales (solo Talking_0..2 rotando)
      const visuals = mapEmotionToVisuals(mia_emocion, nextIndex - 1);

      // TTS + lipsync
      const idx = 0;
      const fileName = `audios/message_${idx}.mp3`;
      await voice.textToSpeech(elevenLabsApiKey, voiceID, fileName, mia_text);
      await lipSyncMessage(idx);
      const audio = await audioFileToBase64(fileName);
      const lipsync = await readJsonTranscript(`audios/message_${idx}.json`);

      // Guardar historial
      historialActual[nextKey] = {
        user_responde: transcript,
        sentimiento,
        mia_emocion,
        mia_text,
      };
      await writeJson(HIST_PATH, historialActual);

      // Actualizar/crear resumen si creció el tamaño
      await maybeUpdateSummary();

      log(`✅ Audio flujo completo generado y guardado como ${nextKey}`);

      // refresca actividad
      lastActivity = Date.now();

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

    // ======= CASO 2: Texto sin audio → flujo original (no tocado salvo utilidades) =======
    if (!userMessage) {
      lastActivity = Date.now();
      return res.send({
        messages: [
          {
            text: "Hey dear... How was your day?",
            audio: await audioFileToBase64("audios/intro_0.wav"),
            lipsync: await readJsonTranscript("audios/intro_0.json"),
            facialExpression: "smile",
            animation: "Talking_1",
          },
        ],
      });
    }

    if (!elevenLabsApiKey || openai.apiKey === "-") {
      lastActivity = Date.now();
      return res.send({
        messages: [
          {
            text: "Please my dear, don't forget to add your API keys!",
            facialExpression: "angry",
            animation: "Angry",
          },
        ],
      });
    }

    // === CHAT NORMAL CON OpenAI + ElevenLabs (tu lógica previa) ===
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo-1106",
      max_tokens: 1000,
      temperature: 0.6,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `
            You are a virtual girlfriend.
            You will always reply with a JSON array of messages. With a maximum of 3 messages.
            Each message has a text, facialExpression, and animation property.
            The different facial expressions are: smile, sad, angry, surprised, funnyFace, and default.
            The different animations are: Talking_0, Talking_1, Talking_2, Crying, Laughing, Rumba, Idle, Terrified, and Angry. 
          `,
        },
        { role: "user", content: userMessage },
      ],
    });

    let messages = JSON.parse(completion.choices[0].message.content);
    if (messages.messages) messages = messages.messages;

    await fs.mkdir("audios", { recursive: true });
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      const fileName = `audios/message_${i}.mp3`;
      await voice.textToSpeech(elevenLabsApiKey, voiceID, fileName, message.text);
      await lipSyncMessage(i);
      message.audio = await audioFileToBase64(fileName);
      message.lipsync = await readJsonTranscript(`audios/message_${i}.json`);
    }

    lastActivity = Date.now();
    res.send({ messages });
  } catch (err) {
    console.error("Error en /chat:", err);
    res.status(500).json({ error: err.message });
  }
});

// ================== Start ==================
app.listen(port, () => {
  console.log(`Virtual Girlfriend listening on port ${port}`);
});
