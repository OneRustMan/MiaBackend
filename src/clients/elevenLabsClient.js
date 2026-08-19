// src/clients/elevenLabsClient.js
import voice from "elevenlabs-node";

// elevenlabs-node se usa con argumentos posicionales y NO soporta cancelación
// (por eso el guard de generación vive en chat.controller.js, antes de llamarlo).
export { voice };
