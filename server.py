import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows-specific fix: Add PyTorch library path to find CUDA DLLs for CTranslate2
if os.name == 'nt':
    import site
    try:
        site_packages = site.getsitepackages()[0]
        torch_lib_path = os.path.join(site_packages, "torch", "lib")
        
        if os.path.exists(torch_lib_path):
            os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
            os.add_dll_directory(torch_lib_path)
            print(f"Loaded CUDA DLLs from: {torch_lib_path}")
        else:
            print("Warning: torch/lib directory not found. Ensure PyTorch with CUDA is installed.")
    except Exception as e:
        print(f"Warning: Error configuring DLL path: {e}")

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from transcriber import WhisperEngine, StreamingSession
from vad import VADEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASR-Backend")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=5)

whisper_engine: WhisperEngine | None = None
vad_engine: VADEngine | None = None  # optional


@app.on_event("startup")
async def startup_event():
    global whisper_engine, vad_engine
    logger.info("Initializing Whisper engine...")
    whisper_engine = WhisperEngine(model_size="large-v2")
    vad_engine = VADEngine()
    logger.info("Engines initialized.")

@app.websocket("/v1/streaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New client connected")

    await websocket.send_json(
        {"message_type": "SessionBegins", "session_id": str(id(websocket))}
    )

    assert whisper_engine is not None
    session = StreamingSession(engine=whisper_engine)

    vad_buffer = np.array([], dtype=np.float32)
    vad_window_sec = 1.0
    vad_min_sec = 0.3

    talking = False
    silence_sec = 0.0
    end_of_utterance_sec = 0.7

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except RuntimeError as e:
                logger.info(f"WebSocket receive error (treat as disconnect): {e}")
                break

            if "bytes" in message:
                audio_chunk: bytes = message["bytes"]

                chunk_f32 = (
                    np.frombuffer(audio_chunk, dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )

                vad_buffer = np.concatenate((vad_buffer, chunk_f32))
                max_vad_samples = int(vad_window_sec * 16000)
                if len(vad_buffer) > max_vad_samples:
                    vad_buffer = vad_buffer[-max_vad_samples:]

                has_speech = False
                if vad_engine is not None and len(vad_buffer) >= int(vad_min_sec * 16000):
                    has_speech = vad_engine.detect_speech(vad_buffer)

                rms = float(np.sqrt(np.mean(chunk_f32 ** 2))) if len(chunk_f32) > 0 else 0.0
                energy_is_speech = rms > 0.01

                chunk_sec = len(chunk_f32) / 16000.0

                if has_speech or energy_is_speech:
                    talking = True
                    silence_sec = 0.0

                    session.add_chunk(audio_chunk)

                    loop = asyncio.get_running_loop()
                    text, is_final = await loop.run_in_executor(
                        executor, session.transcribe_partial
                    )
                    if text:
                        await websocket.send_json(
                            {
                                "message_type": "PartialTranscript",
                                "text": text,
                                "created": "now",
                            }
                        )
                else:
                    if talking:
                        silence_sec += chunk_sec
                        if silence_sec >= end_of_utterance_sec:
                            loop = asyncio.get_running_loop()
                            final_text, _ = await loop.run_in_executor(
                                executor, session.transcribe_full
                            )
                            if final_text:
                                await websocket.send_json(
                                    {
                                        "message_type": "FinalTranscript",
                                        "text": final_text,
                                        "created": "now",
                                    }
                                )

                            session.reset()
                            talking = False
                            silence_sec = 0.0

            elif "text" in message:
                pass

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
