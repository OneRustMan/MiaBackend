Base del proyecto: wass08 <https://github.com/wass08>\

Hice un fork inicial del repositorio <https://github.com/wass08/r3f-virtual-girlfriend-backend>

## Pasos para levantar el backend*

1. Crear archivo .env
Este archivo debe contener los api keys de:
```
OPENAI_API_KEY=sk-...
ELEVEN_LABS_API_KEY=...
```
### OPENAI_API_KEY
- Se utiliza para el TTS (Whisper).
- Se utiliza para formular la respuesta al usuario (gpt-versionx)

### ELEVEN_LABS_API_KEY
- Se utiliza para el STT (TextToSpeech)

2. Instalar la dependencia ffmpeg y rhubarb
Rhubarb
[https://github.com/DanielSWolf/rhubarb-lip-sync/releases]


3. Start the development server with
```
yarn
yarn dev
```
