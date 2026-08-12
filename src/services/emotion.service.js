// src/services/emotion.service.js
import { MODELS_BASE_URL } from "../config/env.js";

// Estas dos lanzan el error hacia arriba a propósito: el call site las envuelve
// con withFallback() para degradar a "neutral"/"default" sin cortar el flujo.
export async function callLocalSentiment(text, signal) {
  const r = await fetch(`${MODELS_BASE_URL}/sentiment`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!r.ok) throw new Error(`sentiment ${r.status}`);
  const j = await r.json();
  return (j.sentimiento || "").toLowerCase();
}

export async function callLocalMiaPredict(text, sentimiento, signal) {
  const r = await fetch(`${MODELS_BASE_URL}/mia_predict`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, sentimiento }),
    signal,
  });
  if (!r.ok) throw new Error(`mia_predict ${r.status}`);
  const j = await r.json();
  return (j.mia_emocion || "").toLowerCase();
}

export function rotatingTalkingAnimation(conversationIndex) {
  const mod = conversationIndex % 3;
  if (mod === 0) return "Talking_0";
  if (mod === 1) return "Talking_1";
  return "Talking_2";
}

export function mapEmotionToVisuals(mia_emocion, conversationIndex) {
  const e = (mia_emocion || "default").toLowerCase();
  let facialExpression = "default";
  if (e === "alegría" || e === "amor") facialExpression = "smile";
  else if (e === "tristeza") facialExpression = "sad";
  else if (e === "ira") facialExpression = "angry";
  else if (e === "miedo" || e === "sorpresa") facialExpression = "surprised";
  return { facialExpression, animation: rotatingTalkingAnimation(conversationIndex) };
}

export function getVoiceSettingsForEmotion(mia_emocion) {
  if (mia_emocion === "alegría") {
    return { stability: 0.3, similarityBoost: 0.75 };
  }
  if (mia_emocion === "amor") {
    return { stability: 0.45, similarityBoost: 0.8 };
  }
  return { stability: 0.5, similarityBoost: 0.75 };
}
