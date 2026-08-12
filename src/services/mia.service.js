// src/services/mia.service.js
import { openai } from "../clients/openaiClient.js";
import { SUMMARY_PATH } from "../config/env.js";
import { readJsonSafe, writeJson } from "../utils/fileStore.js";

export function sanitizeSummary(text) {
  if (!text) return text;
  return text
    .replace(/^\s*\[.*?\]\s*/gm, "")   // saca cualquier [encabezado] al inicio de línea
    .replace(/^\s*[-•]\s*/gm, "")       // saca viñetas sueltas
    .replace(/\n{2,}/g, " ")             // colapsa saltos de línea múltiples
    .replace(/\n/g, " ")                  // colapsa saltos de línea simples
    .trim();
}

export async function updateRollingSummary(latestTurn, signal) {
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
  }, { signal });

  const resumen = sanitizeSummary((resp.output_text || resumenPrevio || "").trim());
  await writeJson(SUMMARY_PATH, {
    resumen,
    updatedAt: new Date().toISOString(),
  });
}

export async function generateMiaReply({ transcript, sentimiento, mia_emocion, signal }) {
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
  }, { signal });

  return (resp.output_text || "").trim();
}
