// src/routes/realtimeChat.routes.js
//
// Equivalente a un Router de express, pero para WebSocket: el upgrade HTTP no
// pasa por el stack de middlewares de express, así que no se puede montar con
// app.use(). Se mantiene el patrón de carpetas igual: acá vive el path, y el
// controller se ocupa de la conexión.

import { WebSocketServer } from "ws";

import { handleRealtimeChatConnection } from "../controllers/realtimeChat.controller.js";

export const REALTIME_CHAT_PATH = "/chat/live";

/**
 * Monta la sesión de chat en vivo sobre el http.Server que devuelve app.listen().
 * `path` hace que el WebSocketServer solo tome los upgrades de /chat/live y deje
 * pasar el resto.
 */
export function mountRealtimeChat(server) {
  const wss = new WebSocketServer({ server, path: REALTIME_CHAT_PATH });
  wss.on("connection", handleRealtimeChatConnection);
  return wss;
}
