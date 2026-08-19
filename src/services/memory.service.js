// src/services/memory.service.js
import { removeStopwords, spa } from "stopword";

import { HIST_PATH } from "../config/env.js";
import { readJsonSafe } from "../utils/fileStore.js";

function normalize(word) {
  return word
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

// Complemento de dominio: palabras funcionales frecuentes en el
// registro conversacional de acompañamiento emocional que la lista
// general "spa" no cubre. Mantener esta lista corta a propósito —
// la base viene de la librería, esto es solo el ajuste específico
// de esta app.
const DOMAIN_STOPWORDS = [
  "empezar", "bastante",
  "esto", "este", "estos", "es", "estan", "estamos", "estar", "estas",
  "estoy",
  "todo", "todos", "otra", "otras", "otros",
  "algo", "algunos", "algunas", "nada", "mucho", "muchos", "solo",
  "tanto", "antes", "ante", "entre",
  "fueron", "eran", "ellos", "ellas", "nosotros", "nosotras",
  "quienes", "unos", "dice", "dijo", "hizo", "hacer",
];

const STOPWORDS_ES = [...spa, ...DOMAIN_STOPWORDS];

// El filtro de longitud > 3 se mantiene como capa extra, además del stopword
// removal del paquete (lista curada de español, código ISO 639-3 "spa").
function extractKeywords(text) {
  if (!text) return [];
  const rawWords = normalize(text)
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 3);
  return removeStopwords(rawWords, STOPWORDS_ES);
}

function wordsOverlap(a, b) {
  if (a === b) return true;
  const [shorter, longer] = a.length <= b.length ? [a, b] : [b, a];
  return shorter.length >= 4 && longer.startsWith(shorter);
}

function getSortedTurnKeys(historial) {
  return Object.keys(historial)
    .filter((k) => k.startsWith("conversacion_"))
    .sort((a, b) => {
      const na = parseInt(a.replace("conversacion_", ""), 10);
      const nb = parseInt(b.replace("conversacion_", ""), 10);
      return na - nb;
    });
}

export async function getRecentTurns(n = 2) {
  const historial = await readJsonSafe(HIST_PATH, {});
  const keys = getSortedTurnKeys(historial);
  const recentKeys = keys.slice(-n);
  return recentKeys.map((k) => ({ key: k, turn: historial[k] }));
}

export async function findRelevantOldTurns(currentTranscript, excludeRecentCount = 2, cap = 2) {
  const historial = await readJsonSafe(HIST_PATH, {});
  const keys = getSortedTurnKeys(historial);
  const oldKeys = keys.slice(0, Math.max(0, keys.length - excludeRecentCount));
  const keywords = extractKeywords(currentTranscript);
  if (keywords.length === 0 || oldKeys.length === 0) return [];

  const matches = oldKeys
    .map((key) => {
      const turn = historial[key];
      const turnWords = extractKeywords(`${turn.user_responde} ${turn.mia_text}`);
      const matchCount = keywords.filter((kw) =>
        turnWords.some((tw) => wordsOverlap(kw, tw))
      ).length;
      const index = parseInt(key.replace("conversacion_", ""), 10);
      return { key, turn, matchCount, index };
    })
    .filter((c) => c.matchCount > 0)
    .sort((a, b) => b.index - a.index)
    .slice(0, cap);

  return matches.map(({ key, turn }) => ({ key, turn }));
}
