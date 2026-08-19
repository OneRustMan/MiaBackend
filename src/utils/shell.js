// src/utils/shell.js
import { exec } from "child_process";

export const execCommand = (command) => new Promise((resolve, reject) => { exec(command, (error, stdout, stderr) => { if (error) { console.error("Command error:", stderr || error.message); return reject(error); } resolve(stdout); }); });
