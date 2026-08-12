// src/clients/openaiClient.js
import OpenAI from "openai";
import { File } from "node:buffer";
import { OPENAI_API_KEY } from "../config/env.js";

if (!globalThis.File) globalThis.File = File;

export const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
