// src/routes/chat.routes.js
import { Router } from "express";
import { handleChat } from "../controllers/chat.controller.js";

const router = Router();

router.post("/", handleChat);

export default router;
