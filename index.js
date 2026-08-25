// index.js
import cors from "cors";
import express from "express";

import { PORT } from "./src/config/env.js";
import { errorHandler } from "./src/middleware/errorHandler.js";
import chatRouter from "./src/routes/chat.routes.js";
import dashboardRouter from "./src/routes/dashboard.routes.js";
import { REALTIME_CHAT_PATH, mountRealtimeChat } from "./src/routes/realtimeChat.routes.js";
import resetRouter from "./src/routes/reset.routes.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "100mb" })); // ok aunque /reset no mande body

app.use("/chat", chatRouter);
app.use("/reset", resetRouter);
app.use("/dashboard", dashboardRouter);

// Tiene que ser lo último que se registra, después de todas las rutas.
app.use(errorHandler);

const server = app.listen(PORT, () => {
  console.log(`Virtual Girlfriend listening on port ${PORT}`);
});

// El chat en vivo va sobre el mismo servidor HTTP: express no maneja upgrades a
// WebSocket, así que se engancha al http.Server que devuelve app.listen().
mountRealtimeChat(server);
console.log(`Live chat WebSocket listening on ws://localhost:${PORT}${REALTIME_CHAT_PATH}`);
