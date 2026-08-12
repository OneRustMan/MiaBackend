// src/controllers/reset.controller.js
import { ensureDirs } from "../services/historial.service.js";
import { resetSession } from "../services/session.service.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { log } from "../utils/logger.js";

export const handleReset = asyncHandler(async (req, res) => {
  await ensureDirs();
  log(`↩️  RESET recibido. query.ts=${req.query?.ts || ""} body=${JSON.stringify(req.body || {})}`);
  await resetSession(req.body?.reason || "manual");
  return res.json({ ok: true, reset: true });
});
