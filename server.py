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
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from transcriber import WhisperEngine, StreamingSession
from vad import VADEngine
from translator import TranslatorEngine, LANG_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASR-Backend")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
translate_executor = ThreadPoolExecutor(max_workers=2)

whisper_engine: WhisperEngine | None = None
vad_engine: VADEngine | None = None
translator_engine: TranslatorEngine | None = None

# Store active ESP session for broadcasting to browser
active_esp_connection: WebSocket | None = None
last_esp_display_fingerprint: str | None = None

# Shared runtime config controlled by WebUI.
runtime_translate = False
runtime_target_lang = "vi"
runtime_context_max = 6
runtime_source_lang = "auto"
runtime_esp_display_mode = "original"
runtime_display_fallback = "error"


@app.on_event("startup")
async def startup_event():
    global whisper_engine, vad_engine, translator_engine
    logger.info("Initializing Whisper engine...")
    whisper_engine = WhisperEngine(model_size="large-v3", compute_type="bfloat16")
    vad_engine = VADEngine()
    logger.info("Whisper + VAD initialized. Loading translator...")
    translator_engine = TranslatorEngine()
    logger.info("All engines initialized and ready.")


@app.websocket("/v1/streaming")
async def websocket_endpoint(websocket: WebSocket):
    global active_esp_connection
    global last_esp_display_fingerprint
    global runtime_translate, runtime_target_lang, runtime_context_max
    global runtime_source_lang, runtime_esp_display_mode, runtime_display_fallback
    
    await websocket.accept()
    logger.info("New client connected")

    # Detect client type from first message
    client_type = "unknown"
    is_esp_client = False

    try:
        first_msg = await asyncio.wait_for(websocket.receive(), timeout=2.0)
        if "text" in first_msg:
            try:
                msg_data = json.loads(first_msg["text"])
                client_type = msg_data.get("client_type", "browser")
                is_esp_client = (client_type == "esp")
            except:
                client_type = "browser"
    except:
        client_type = "browser"

    logger.info(f"Client connected: type={client_type}")

    # Store ESP connection for broadcasting
    if is_esp_client:
        active_esp_connection = websocket

    await websocket.send_json(
        {"message_type": "SessionBegins", "session_id": str(id(websocket)), "client_type": client_type}
    )

    assert whisper_engine is not None
    session = StreamingSession(engine=whisper_engine)

    # VAD uses direct model() call on each chunk - fast single-pass probability
    talking = False
    silence_sec = 0.0
    end_of_utterance_sec = 0.3

    # Per-session context cache for translation quality
    context_history: list[str] = []
    utterance_id = 0

    def reset_stream_state(clear_context: bool = False):
        nonlocal talking, silence_sec
        session.reset()
        talking = False
        silence_sec = 0.0
        if vad_engine is not None:
            vad_engine.reset_state()
        if clear_context:
            context_history.clear()

    async def send_to_active_esp(payload: dict):
        global active_esp_connection
        if active_esp_connection is None:
            return
        try:
            await active_esp_connection.send_json(payload)
        except Exception:
            # Drop stale connection references if send fails.
            active_esp_connection = None

    async def push_display_to_esp(text: str, mode: str):
        global last_esp_display_fingerprint
        if not text:
            return
        fingerprint = f"{mode}:{text}"
        if fingerprint == last_esp_display_fingerprint:
            return
        await send_to_active_esp({
            "message_type": "DisplayText",
            "text": text,
            "mode": mode,
        })
        last_esp_display_fingerprint = fingerprint

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                if is_esp_client:
                    active_esp_connection = None
                break
            except RuntimeError as e:
                logger.info(f"WebSocket receive error (treat as disconnect): {e}")
                if is_esp_client:
                    active_esp_connection = None
                break

            if "bytes" in message:
                audio_chunk: bytes = message["bytes"]

                # Single int16->float32 conversion (reused for both VAD and buffer)
                chunk_f32 = (
                    np.frombuffer(audio_chunk, dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )

                # Fast VAD: direct model() probability check on chunk
                has_speech = False
                if vad_engine is not None and len(chunk_f32) >= 512:
                    has_speech = vad_engine.detect_speech(chunk_f32)

                chunk_sec = len(chunk_f32) / 16000.0

                if has_speech:
                    if not talking:
                        talking = True
                        # Notify current client that speech started
                        await websocket.send_json(
                            {"message_type": "SpeechDetected"}
                        )

                    silence_sec = 0.0

                    # Buffer audio (float32 directly - no double conversion)
                    session.add_chunk_f32(chunk_f32)
                else:
                    if talking:
                        silence_sec += chunk_sec
                        if silence_sec >= end_of_utterance_sec:
                            # === Single Whisper call per utterance ===
                            loop = asyncio.get_running_loop()
                            final_text, _ = await loop.run_in_executor(
                                executor, session.transcribe_full
                            )
                            if final_text:
                                utterance_id += 1
                                translated_text: str | None = None
                                # Always send FinalTranscript to the connected client
                                await websocket.send_json(
                                    {
                                        "message_type": "FinalTranscript",
                                        "text": final_text,
                                        "utterance_id": utterance_id,
                                        "created": "now",
                                    }
                                )

                                # If translation enabled, send TranslatedTranscript to connected client
                                if runtime_translate and translator_engine is not None:
                                    ctx_str = "\n".join(context_history[-runtime_context_max:]) if context_history else None
                                    try:
                                        translated_text = await loop.run_in_executor(
                                            translate_executor,
                                            translator_engine.translate,
                                            final_text,
                                            runtime_target_lang,
                                            ctx_str,
                                        )
                                        if translated_text:
                                            await websocket.send_json(
                                                {
                                                    "message_type": "TranslatedTranscript",
                                                    "text": translated_text,
                                                    "original": final_text,
                                                    "utterance_id": utterance_id,
                                                }
                                            )
                                    except Exception as te:
                                        logger.error(f"Translation error: {te}\n{traceback.format_exc()}")
                                    context_history.append(final_text)
                                    if len(context_history) > runtime_context_max:
                                        context_history = context_history[-runtime_context_max:]

                                # Always update OLED from server-selected output (independent from mic source).
                                if runtime_esp_display_mode == "translated":
                                    if runtime_translate and translated_text:
                                        await push_display_to_esp(translated_text, "translated")
                                    elif runtime_translate and runtime_display_fallback == "error":
                                        await push_display_to_esp("[translate unavailable]", "error")
                                    elif runtime_translate:
                                        await push_display_to_esp(final_text, "original")
                                    else:
                                        await push_display_to_esp("[translation off]", "error")
                                else:
                                    await push_display_to_esp(final_text, "original")

                            reset_stream_state()

                            # Notify client that speech ended
                            await websocket.send_json(
                                {"message_type": "SilenceDetected"}
                            )

            elif "text" in message:
                try:
                    msg = json.loads(message["text"])

                    # Safe dynamic client classification: allow upgrade from
                    # unknown/browser to ESP when explicit marker arrives late.
                    if "client_type" in msg and not is_esp_client:
                        reported_type = msg.get("client_type")
                        if reported_type == "esp" and client_type != "esp":
                            logger.info(f"Client classification updated: {client_type} -> esp")
                            client_type = "esp"
                            is_esp_client = True
                            active_esp_connection = websocket

                    msg_type = msg.get("type")
                    if msg_type == "ping":
                        logger.debug(f"Ping received: t={msg.get('t')}")
                        await websocket.send_json({
                            "message_type": "Pong",
                            "t": msg.get("t", 0),
                            "server_time": int(time.time() * 1000),
                        })
                    elif msg_type == "config":
                        runtime_translate = msg.get("translate", False)
                        requested_target = msg.get("target_lang", "vi")
                        runtime_target_lang = requested_target if requested_target in LANG_MAP else "vi"

                        requested_context = msg.get("context_max", 6)
                        try:
                            runtime_context_max = int(requested_context)
                        except (TypeError, ValueError):
                            runtime_context_max = 6
                        runtime_context_max = min(max(runtime_context_max, 1), 12)

                        context_history.clear()

                        # ESP display mode: "original" = thô, "translated" = đã dịch
                        requested_mode = msg.get("esp_display_mode", "original")
                        runtime_esp_display_mode = requested_mode if requested_mode in ["original", "translated"] else "original"
                        requested_fallback = msg.get("display_fallback", "error")
                        runtime_display_fallback = requested_fallback if requested_fallback in ["error", "original", "skip"] else "error"

                        # Language pinning: skip auto-detect if source_lang set
                        runtime_source_lang = msg.get("source_lang", "auto")
                        whisper_engine.set_language(
                            None if runtime_source_lang == "auto" else runtime_source_lang
                        )

                        # Sync runtime display settings to physical ESP client.
                        if not is_esp_client:
                            await send_to_active_esp({
                                "message_type": "RuntimeConfig",
                                "esp_display_mode": runtime_esp_display_mode,
                                "translate": runtime_translate,
                                "target_lang": runtime_target_lang,
                                "source_lang": runtime_source_lang,
                                "context_max": runtime_context_max,
                                "display_fallback": runtime_display_fallback,
                            })

                        logger.info(
                            f"Config updated: translate={runtime_translate}, "
                            f"target_lang={runtime_target_lang}, source_lang={runtime_source_lang}, "
                            f"context_max={runtime_context_max}, esp_display_mode={runtime_esp_display_mode}, "
                            f"display_fallback={runtime_display_fallback}"
                        )

                        await websocket.send_json({
                            "message_type": "ConfigAck",
                            "translate": runtime_translate,
                            "target_lang": runtime_target_lang,
                            "context_max": runtime_context_max,
                            "source_lang": runtime_source_lang,
                            "esp_display_mode": runtime_esp_display_mode,
                            "display_fallback": runtime_display_fallback,
                        })
                    elif msg_type == "esp_start":
                        if not is_esp_client:
                            await send_to_active_esp({"message_type": "esp_start"})
                            logger.info("Browser requested ESP audio streaming start")
                        else:
                            logger.info("ESP requested to start audio streaming")
                    elif msg_type == "esp_stop":
                        if not is_esp_client:
                            await send_to_active_esp({"message_type": "esp_stop"})
                            logger.info("Browser requested ESP audio streaming stop")
                        else:
                            logger.info("ESP requested to stop audio streaming")
                    elif msg_type == "reset_session":
                        reset_stream_state(clear_context=False)
                        await websocket.send_json({"message_type": "SessionReset"})
                except (json.JSONDecodeError, TypeError):
                    pass

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        if is_esp_client:
            active_esp_connection = None
