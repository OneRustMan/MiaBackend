// src/services/mia.service.js
import { openai } from "../clients/openaiClient.js";
import { HIST_PATH, OPENAI_TEXT_MODEL, SUMMARY_PATH } from "../config/env.js";
import { readJsonSafe, writeJson } from "../utils/fileStore.js";

// El prompt ya pide 220 caracteres, pero el modelo no siempre respeta el límite:
// este es el techo duro que garantiza que el resumen no crezca turno a turno.
const MAX_SUMMARY_CHARS = 220;

// Conserva el FINAL del texto, no el principio: el modelo escribe siempre el
// turno más reciente al final del párrafo, y eso es lo que generateMiaReply
// más necesita. Lo que se descarta es el contexto viejo del principio.
function enforceSummaryLimit(text) {
  if (!text || text.length <= MAX_SUMMARY_CHARS) return text;
  const truncated = text.slice(-MAX_SUMMARY_CHARS);
  const firstSpace = truncated.indexOf(" ");
  const clean = (firstSpace > 0 ? truncated.slice(firstSpace + 1) : truncated).trim();
  return "…" + clean;
}

export function sanitizeSummary(text) {
  if (!text) return text;
  return text
    .replace(/^\s*\[.*?\]\s*/gm, "")   // saca cualquier [encabezado] al inicio de línea
    .replace(/^\s*[-•]\s*/gm, "")       // saca viñetas sueltas
    .replace(/\n{2,}/g, " ")             // colapsa saltos de línea múltiples
    .replace(/\n/g, " ")                  // colapsa saltos de línea simples
    .trim();
}

// El resumen se actualiza en background (fire-and-forget desde el controller),
// así que puede haber uno todavía corriendo cuando llega el turno siguiente.
// Guardamos esa promesa para poder encadenar updates (nunca dos a la vez
// pisándose el archivo) y para que el turno siguiente pueda esperarla justo
// antes de leer el resumen (waitForPendingSummary).
let pendingSummaryPromise = null;

export function updateRollingSummary(latestTurn, signal) {
  const previous = pendingSummaryPromise
    ? pendingSummaryPromise.catch(() => {})
    : Promise.resolve();
  const current = previous.then(() => runUpdateRollingSummary(latestTurn, signal));
  pendingSummaryPromise = current;
  return current;
}

// Resuelve al instante si no hay ningún update en vuelo (primer turno).
// El .catch() acá evita que un error del turno anterior se filtre al siguiente:
// ese error ya se reporta en el .catch() del fire-and-forget del controller.
export async function waitForPendingSummary() {
  if (pendingSummaryPromise) {
    await pendingSummaryPromise.catch(() => {});
  }
}

async function runUpdateRollingSummary(latestTurn, signal) {
  const summaryCurrent = await readJsonSafe(SUMMARY_PATH, {});
  const resumenPrevio = summaryCurrent?.resumen || "";

  const system = `
Eres un sistema que mantiene un resumen narrativo compacto de una conversación de acompañamiento emocional entre un usuario y MIA.

Reglas estrictas que debés seguir siempre:
- Reescribí el resumen completo desde cero, integrando lo nuevo. NUNCA agregues el turno nuevo como una línea extra al final del resumen anterior — sintetizá todo junto en un texto nuevo y más corto si hace falta.
- Formato: un único párrafo en prosa corrida. Sin viñetas, sin guiones, sin títulos, sin corchetes, sin las etiquetas "Usuario:" o "MIA:".
- Extensión objetivo: 160 caracteres en total (nunca más de 200). Si ya estás cerca, comprimí agresivamente en vez de seguir agregando.
- Si tenés que elegir qué recortar para cumplir la extensión, sacrificá primero detalle de turnos VIEJOS (resumilos en pocas palabras o mencionalos de pasada). El turno MÁS RECIENTE — lo último que pasó — tiene que quedar siempre completo y con sus datos concretos intactos (nombres, hechos, estados), aunque el resto del resumen quede muy comprimido.
- No uses frases introductorias como "Nuevo turno:", "En el turno más reciente," o similares — arrancá directo con el contenido, en prosa corrida.
- Parafraseá el contenido y el estado emocional; no cites frases textuales de lo que dijeron.
- Tu respuesta completa ES el resumen y nada más. No incluyas encabezados ni texto explicativo.
- Escribí en español, sin emojis.

Ejemplo de formato correcto:
"El usuario suele saludar con ánimo alegre; en el turno más reciente contó que perdió a su gato y MIA validó su tristeza ofreciéndose a escuchar."

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
    model: OPENAI_TEXT_MODEL,
    reasoning: { effort: "minimal" },
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  }, { signal });

  const resumen = enforceSummaryLimit(sanitizeSummary((resp.output_text || resumenPrevio || "").trim()));
  await writeJson(SUMMARY_PATH, {
    resumen,
    updatedAt: new Date().toISOString(),
  });
}

export async function generateMiaReply({ transcript, sentimiento, mia_emocion, signal }) {
  const summary = await readJsonSafe(SUMMARY_PATH, {});
  const resumen = summary?.resumen || "";

  // El resumen puede truncar y perder el turno más reciente, así que ese turno
  // lo leemos completo del historial (que se guarda íntegro, sin truncar).
  const historial = await readJsonSafe(HIST_PATH, {});
  const claves = Object.keys(historial).filter(k => k.startsWith("conversacion_"));
  const ultimaClave = claves.sort((a, b) => {
    const na = parseInt(a.replace("conversacion_", ""), 10);
    const nb = parseInt(b.replace("conversacion_", ""), 10);
    return na - nb;
  }).at(-1);
  const ultimoTurno = ultimaClave ? historial[ultimaClave] : null;

  const system = `
Eres "MIA", un agente de IA empática y de acompañamiento emocional.
Respondes SIEMPRE en español, con calidez y claridad, sin emojis.
`.trim();

  const user = `
[Resumen de la conversación hasta ahora]
${resumen || "(sin resumen aún, es el primer turno)"}

[Último intercambio completo]
${ultimoTurno ? `Usuario dijo: "${ultimoTurno.user_responde}"\nMIA respondió: "${ultimoTurno.mia_text}"` : "(no hay turno anterior, es el primer turno)"}

[Turno actual]
${transcript}

[Metadatos]
- Sentimiento del usuario: ${sentimiento}
- Emoción de respuesta: ${mia_emocion}

[Instrucciones]
- Breve (2 frases), valida y acompaña. Sin consejos clínicos.
`.trim();

  const resp = await openai.responses.create({
    model: OPENAI_TEXT_MODEL,
    reasoning: { effort: "minimal" },
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  }, { signal });

  return (resp.output_text || "").trim();
}
