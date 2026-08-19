// src/utils/asyncHandler.js
// Envuelve un handler async para que cualquier rechazo caiga en next(err)
// y lo maneje el error handler global en vez de quedar como unhandled rejection.
export const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);
