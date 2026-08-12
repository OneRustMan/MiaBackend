// src/services/historial.service.js
import { promises as fs } from "fs";
import { HIST_DIR, HIST_PATH, SUMMARY_PATH } from "../config/env.js";
import { readJsonSafe, writeJson } from "../utils/fileStore.js";

export async function ensureDirs() {
  await fs.mkdir(HIST_DIR, { recursive: true });
  await fs.mkdir("audios", { recursive: true });
}

export async function clearHistorial() {
  await fs.mkdir(HIST_DIR, { recursive: true });
  await writeJson(HIST_PATH, {});
  await writeJson(SUMMARY_PATH, {});
}

export async function readHistorial() {
  return readJsonSafe(HIST_PATH, {});
}

export async function writeHistorial(historial) {
  await writeJson(HIST_PATH, historial);
}
