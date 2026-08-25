// src/utils/logger.js
export function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

// Instrumentación temporal: mide cada paso secuencial de un turno para saber
// dónde se va el tiempo. No altera el flujo. Vive acá porque lo usan tanto el
// controller (transcripción) como turnPipeline.service.js (el resto del turno).
export function logStep(label, startedAt) {
  log(`⏱️  ${label}: ${Date.now() - startedAt}ms`);
}
