// src/services/mia.service.js
import { openai } from "../clients/openaiClient.js";
import { OPENAI_TEXT_MODEL } from "../config/env.js";
import { findRelevantOldTurns, getRecentTurns } from "./memory.service.js";

// Pedirle por prompt que "no repita frases de cierre" no alcanza: el modelo
// vuelve a caer en la misma fórmula ("estoy aquí a tu lado para lo que
// necesites") turno tras turno. Igual que rotatingTalkingAnimation con las
// animaciones, acá le sacamos la decisión al modelo: el turno determina qué
// tipo de cierre le toca, y esa instrucción concreta entra al prompt.
const CLOSING_STYLES = [
  "Cerrá con una pregunta abierta sobre cómo se siente, sin usar frases de cierre genéricas.",
  "Cerrá con una validación breve de lo que sintió, sin hacer pregunta y sin fórmulas de acompañamiento tipo 'estoy aquí a tu lado'.",
  "Cerrá retomando un detalle CONCRETO de lo que acaba de contar (un nombre, un lugar, un hecho puntual que haya mencionado) y quedate ahí, sin invitar explícitamente a seguir hablando. Ejemplo del tono: en vez de 'cuéntame más sobre cómo te sentiste', algo como 'ese primer kilómetro después de tanto tiempo debe haberse sentido enorme'.",
  "Cerrá con una frase que nombre algo específico y personal de lo que contó (un nombre propio, un detalle concreto de la situación), en vez de una frase de acompañamiento genérica. Ejemplo del tono: en vez de 'estoy aquí para acompañarte', algo como 'lo de Rocco recién diagnosticado pesa distinto cuando es tan de golpe' o 'tres meses sin ver a tus amigos se nota en cómo lo contás'.",
];

// turnIndex viene 1-based desde el controller (nextIndex).
function pickClosingStyle(turnIndex) {
  return CLOSING_STYLES[(turnIndex - 1) % CLOSING_STYLES.length];
}

function formatTurn(turn) {
  return `Usuario dijo: "${turn.user_responde}"\nMIA respondió: "${turn.mia_text}"`;
}

export async function generateMiaReply({ transcript, sentimiento, mia_emocion, turnIndex, signal }) {
  // Sin resumen por IA: los 2 últimos turnos van completos, y de los más viejos
  // solo entran los que el turno actual menciona (búsqueda por palabras clave).
  const recentTurns = await getRecentTurns(2);
  const relevantOld = await findRelevantOldTurns(transcript, 2, 2);

  const closingStyle = pickClosingStyle(turnIndex);

  const system = `
Eres "MIA", un agente de IA empática y de acompañamiento emocional.
Respondes SIEMPRE en español, con calidez y claridad, sin emojis.
`.trim();

  // Las secciones se arman como bloques y se unen filtrando los vacíos: si no
  // hay turnos viejos relacionados, esa sección no existe en el prompt (ni
  // siquiera como encabezado suelto o línea en blanco de más).
  const bloques = [
    `[Turnos recientes]\n${recentTurns.length === 0
      ? "(sin turnos previos, es el primer turno)"
      : recentTurns.map(({ turn }) => formatTurn(turn)).join("\n---\n")}`,

    relevantOld.length > 0
      ? `[Turnos anteriores relacionados con lo que acaba de decir]\n${relevantOld.map(({ turn }) => formatTurn(turn)).join("\n---\n")}`
      : null,

    `[Turno actual]\n${transcript}`,

    `[Metadatos]\n- Sentimiento del usuario: ${sentimiento}\n- Emoción de respuesta: ${mia_emocion}`,

    `[Instrucciones]\n- Breve (2 frases). Sin consejos clínicos.\n- ${closingStyle}`,
  ];

  const user = bloques.filter(Boolean).join("\n\n");

  const resp = await openai.responses.create({
    model: OPENAI_TEXT_MODEL,
    reasoning: { effort: "low" },
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  }, { signal });

  return (resp.output_text || "").trim();
}

// Variante en streaming del mismo prompt: en vez de esperar el texto completo,
// va cortando por frase a medida que llegan los deltas, para que el controller
// pueda generar audio de cada frase mientras el modelo sigue escribiendo.
export async function* generateMiaReplyStream({ transcript, sentimiento, mia_emocion, turnIndex, signal }) {
  const recentTurns = await getRecentTurns(2);
  const relevantOld = await findRelevantOldTurns(transcript, 2, 2);

  const closingStyle = pickClosingStyle(turnIndex);

  const system = `
Eres "MIA", un agente de IA empática y de acompañamiento emocional.
Respondes SIEMPRE en español, con calidez y claridad, sin emojis.
`.trim();

  const bloques = [
    `[Turnos recientes]\n${recentTurns.length === 0
      ? "(sin turnos previos, es el primer turno)"
      : recentTurns.map(({ turn }) => formatTurn(turn)).join("\n---\n")}`,

    relevantOld.length > 0
      ? `[Turnos anteriores relacionados con lo que acaba de decir]\n${relevantOld.map(({ turn }) => formatTurn(turn)).join("\n---\n")}`
      : null,

    `[Turno actual]\n${transcript}`,

    `[Metadatos]\n- Sentimiento del usuario: ${sentimiento}\n- Emoción de respuesta: ${mia_emocion}`,

    `[Instrucciones]\n- Breve (2 frases). Sin consejos clínicos.\n- ${closingStyle}`,
  ];

  const user = bloques.filter(Boolean).join("\n\n");

  const stream = await openai.responses.create({
    model: OPENAI_TEXT_MODEL,
    reasoning: { effort: "minimal" },
    stream: true,
    input: [
      { role: "system", content: [{ type: "input_text", text: system }] },
      { role: "user",   content: [{ type: "input_text", text: user }] },
    ],
  }, { signal });

  let buffer = "";
  const SENTENCE_END = /[.!?](?:\s|$)/;

  // El prompt pide 2 frases, pero el modelo se desborda a 3-4 cada tanto. Acá
  // cada frase de más es una llamada extra a ElevenLabs, así que el límite se
  // aplica en código y no se espera a que el modelo decida parar solo.
  const MAX_SENTENCES = 2;
  let sentenceCount = 0;

  for await (const event of stream) {
    if (event.type !== "response.output_text.delta") continue;
    buffer += event.delta;
    // Puede llegar más de una frase en el mismo delta, así que vaciamos el
    // buffer en un while y no con un solo corte por evento.
    let match;
    while ((match = SENTENCE_END.exec(buffer))) {
      const cutAt = match.index + 1;
      const sentence = buffer.slice(0, cutAt).trim();
      buffer = buffer.slice(cutAt);
      if (sentence) {
        yield sentence;
        sentenceCount++;
        // El return corta el generador acá mismo, sin pasar por el leftover:
        // no queremos una tercera frase parcial colgando después del límite.
        if (sentenceCount >= MAX_SENTENCES) return;
      }
    }
  }

  // Última frase sin puntuación de cierre: igual hay que decirla.
  const leftover = buffer.trim();
  if (leftover) yield leftover;
}
