// src/middleware/errorHandler.js
import OpenAI from "openai";
import { log } from "../utils/logger.js";

// Express reconoce un error handler por la aridad de 4 argumentos:
// `next` tiene que estar declarado aunque no se use.
// eslint-disable-next-line no-unused-vars
export function errorHandler(err, req, res, next) {
  // Cancelación por /reset. Hay dos formas según quién lanzó:
  // - fetch nativo (sentiment / mia_predict) → DOMException con name "AbortError"
  // - SDK de OpenAI → APIUserAbortError (su `name` NO es "AbortError")
  // Va ANTES del chequeo de APIError porque APIUserAbortError hereda de APIError.
  if (err.name === "AbortError" || err instanceof OpenAI.APIUserAbortError) {
    log(`⚠️ Chat cancelado por reset (gen ${req.sessionGeneration ?? "?"}).`);
    return res.status(200).json({ ok: false, aborted: true, messages: [] });
  }

  if (err instanceof OpenAI.APIError) {
    console.error(`Error de OpenAI en ${req.baseUrl} [status=${err.status} code=${err.code}]:`, err.message);
    return res.status(502).json({ ok: false, error: "Falló el servicio de IA, intentá de nuevo.", messages: [] });
  }

  console.error(`Error en ${req.baseUrl}:`, err);
  return res.status(500).json({ ok: false, error: err.message, messages: [] });
}
