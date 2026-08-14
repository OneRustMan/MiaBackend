// src/routes/dashboard.routes.js
import { Router } from "express";
import { handleDashboardStream } from "../controllers/dashboard.controller.js";

const router = Router();

router.get("/stream", handleDashboardStream);

export default router;
