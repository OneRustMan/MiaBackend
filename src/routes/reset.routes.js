// src/routes/reset.routes.js
import { Router } from "express";
import { handleReset } from "../controllers/reset.controller.js";

const router = Router();

router.post("/", handleReset);

export default router;
