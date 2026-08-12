// src/services/audio.service.js
import { promises as fs } from "fs";
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
