// src/utils/withFallback.js
// "Probá esto; si falla, logueá un warning y seguí con un valor de reemplazo."
export async function withFallback(fn, fallbackValue, label) {
  try { return await fn(); }
  catch (e) { console.warn(`${label}:`, e.message); return fallbackValue; }
}
