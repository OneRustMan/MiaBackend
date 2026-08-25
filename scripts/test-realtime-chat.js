// scripts/test-realtime-chat.js
//
// Prueba end-to-end del chat en vivo: audio → transcripción por Realtime API →
// pipeline de MIA → respuesta con texto y audio, todo por el mismo WebSocket.
//
//   node scripts/test-realtime-chat.js [--audio <archivo>] [--chunk-ms 100]
//                                      [--url ws://localhost:3000/chat/live]
//                                      [--realtime] [--no-commit]
//
// El audio se convierte con ffmpeg a lo único que acepta la API: PCM 16-bit
// little-endian, mono, 24 kHz. Si el archivo ya es .pcm/.raw se manda tal cual.
//
// El commit del turno es FIJO acá: se manda apenas termina el audio. No hay corte
// automático de turno en ninguno de los dos modelos vigentes, y la detección de
// silencio del lado del cliente es trabajo del frontend (fase 3).
//
// Requiere el backend levantado (yarn dev) — este script NO habla con OpenAI
// directamente, todo pasa por el backend.

import { spawn } from "child_process";
import { readFile } from "fs/promises";
import path from "path";
import WebSocket from "ws";

const SAMPLE_RATE = 24000;
const BYTES_PER_SAMPLE = 2;

function parseArgs(argv) {
  const args = {
    audio: "assets/greeting/greeting.wav",
    chunkMs: 100,
    url: "ws://localhost:3000/chat/live",
    realtime: false,
    commit: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--audio") args.audio = argv[++i];
    else if (a === "--chunk-ms") args.chunkMs = Number(argv[++i]);
    else if (a === "--url") args.url = argv[++i];
    else if (a === "--realtime") args.realtime = true;
    else if (a === "--no-commit") args.commit = false;
    else { console.error(`Argumento desconocido: ${a}`); process.exit(1); }
  }
  return args;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ffmpeg a stdout: PCM crudo, sin header WAV.
function toPcm24k(file) {
  return new Promise((resolve, reject) => {
    const ff = spawn("ffmpeg", [
      "-hide_banner", "-loglevel", "error",
      "-i", file,
      "-f", "s16le", "-acodec", "pcm_s16le",
      "-ar", String(SAMPLE_RATE), "-ac", "1",
      "pipe:1",
    ]);
    const out = [];
    const err = [];
    ff.stdout.on("data", (d) => out.push(d));
    ff.stderr.on("data", (d) => err.push(d));
    ff.on("error", reject);
    ff.on("close", (code) => {
      if (code !== 0) reject(new Error(`ffmpeg salió con código ${code}: ${Buffer.concat(err)}`));
      else resolve(Buffer.concat(out));
    });
  });
}

async function loadAudio(file) {
  const ext = path.extname(file).toLowerCase();
  if (ext === ".pcm" || ext === ".raw") return readFile(file);
  return toPcm24k(file);
}

const args = parseArgs(process.argv.slice(2));

const pcm = await loadAudio(args.audio);
const durationSec = pcm.length / (SAMPLE_RATE * BYTES_PER_SAMPLE);
const chunkBytes = Math.floor((SAMPLE_RATE * BYTES_PER_SAMPLE * args.chunkMs) / 1000);

console.log(`🎧 Audio     : ${args.audio}`);
console.log(`   PCM       : ${pcm.length} bytes (${durationSec.toFixed(2)}s @ ${SAMPLE_RATE}Hz mono)`);
console.log(`   Chunks    : ${Math.ceil(pcm.length / chunkBytes)} de ${args.chunkMs}ms`);
console.log(`   Envío     : ${args.realtime ? "a ritmo real (con pausas)" : "lo más rápido posible"}`);
console.log(`🔌 Conectando a ${args.url}...\n`);

const ws = new WebSocket(args.url);

let tCommit = 0;
let firstChunkAt = 0;
let chunkCount = 0;
let fullText = "";
let audioBytes = 0;
let exitCode = 1;

// Red de seguridad: si el backend nunca contesta, no dejamos el proceso colgado.
const guard = setTimeout(() => {
  console.error("\n⏰ Timeout: no llegó el evento final en 90s.");
  try { ws.close(); } catch {}
  process.exit(1);
}, 90000);

function finish(code) {
  exitCode = code;
  clearTimeout(guard);
  try { ws.close(1000, "fin de la prueba"); } catch {}
}

async function streamAudio() {
  for (let off = 0; off < pcm.length; off += chunkBytes) {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(pcm.subarray(off, off + chunkBytes));
    if (args.realtime) await sleep(args.chunkMs);
  }
  console.log(`📤 Audio enviado completo (${pcm.length} bytes).`);

  if (!args.commit) {
    console.log("⏸️  --no-commit: no se cierra el turno (el servidor queda esperando).");
    return;
  }
  // Commit fijo apenas termina el audio: mecanismo manual de esta fase.
  tCommit = Date.now();
  ws.send(JSON.stringify({ type: "commit" }));
  console.log("✂️  Commit de turno enviado. Esperando respuesta de MIA...\n");
}

ws.on("open", () => console.log("✅ WebSocket abierto."));

ws.on("message", (raw) => {
  let event;
  try {
    event = JSON.parse(raw.toString());
  } catch {
    console.log("⬅️  (mensaje no-JSON)", raw.length, "bytes");
    return;
  }

  switch (event.type) {
    case "ready":
      console.log(`🟢 Sesión de transcripción lista (modelo ${event.model}, id ${event.sessionId}).`);
      streamAudio();
      break;

    case "transcript.delta":
      process.stdout.write(event.text);
      break;

    case "transcript":
      console.log(`\n📝 Transcripción final: "${event.text}"`);
      if (tCommit) console.log(`   (${Date.now() - tCommit}ms desde el commit)`);
      break;

    case "skipped":
      console.log(`⚠️  Turno omitido: ${event.reason}`);
      finish(1);
      break;

    case "chunk": {
      chunkCount++;
      if (!firstChunkAt) {
        firstChunkAt = Date.now();
        if (tCommit) console.log(`⚡ Primer chunk a los ${firstChunkAt - tCommit}ms del commit.`);
      }
      fullText += (fullText ? " " : "") + event.text;
      const b64 = event.audio ? event.audio.length : 0;
      audioBytes += b64;
      const cues = event.lipsync?.mouthCues?.length ?? 0;
      console.log(`💬 chunk ${chunkCount}: "${event.text}"`);
      console.log(`   audio: ${b64 ? `${b64} chars base64 (~${Math.round(b64 * 0.75 / 1024)} KB mp3)` : "SIN AUDIO"} | lipsync: ${cues} cues | ${event.facialExpression} / ${event.animation}`);
      break;
    }

    case "done":
      console.log(`\n✅ done — ${chunkCount} chunks, ${audioBytes} chars de audio base64 en total.`);
      console.log(`   Respuesta completa de MIA: "${fullText}"`);
      if (tCommit) console.log(`   Turno completo en ${Date.now() - tCommit}ms desde el commit.`);
      finish(0);
      break;

    case "aborted":
      console.log("\n⚠️  aborted — el turno se canceló por un /reset de sesión.");
      finish(0);
      break;

    case "error":
      console.error(`\n❌ error del servidor: ${event.error}`);
      finish(1);
      break;

    default:
      console.log("⬅️ ", JSON.stringify(event));
  }
});

ws.on("error", (err) => {
  console.error("❌ Error de WebSocket:", err.message);
  clearTimeout(guard);
  process.exit(1);
});

ws.on("close", (code, reason) => {
  console.log(`🔌 WebSocket cerrado (${code} ${reason?.toString() || ""}).`);
  clearTimeout(guard);
  process.exit(exitCode);
});
