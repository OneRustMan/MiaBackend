// src/services/dashboard.service.js
// Broadcast SSE hacia dashboards conectados (solo lectura, para la demo en feria).
const clients = new Set();

export function registerDashboardClient(res) {
  clients.add(res);
}

export function unregisterDashboardClient(res) {
  clients.delete(res);
}

export function broadcastTurn(data) {
  const payload = `data: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) {
    res.write(payload);
  }
}

export function broadcastReset() {
  broadcastTurn({ type: "reset" });
}
