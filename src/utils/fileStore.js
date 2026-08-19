// src/utils/fileStore.js
import { promises as fs } from "fs";

// Este try/catch es propio de readJsonSafe (archivo inexistente o JSON corrupto
// → fallback silencioso) y NO se reemplaza por withFallback: son cosas distintas.
export async function readJsonSafe(filePath, fallback = {}) {
  try { return JSON.parse(await fs.readFile(filePath, "utf8")); }
  catch { return fallback; }
}

export async function writeJson(filePath, obj) {
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2));
}
