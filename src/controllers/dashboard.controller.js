// src/controllers/dashboard.controller.js
import { registerDashboardClient, unregisterDashboardClient } from "../services/dashboard.service.js";

// Sin asyncHandler: no es async ni responde-y-termina, mantiene la conexión
// SSE abierta hasta que el cliente la cierre.
export function handleDashboardStream(req, res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
  });
  res.write(": conectado\n\n");
  registerDashboardClient(res);

  req.on("close", () => {
    unregisterDashboardClient(res);
  });
}
