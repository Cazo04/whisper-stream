import asyncio
import json
import logging
import signal
import sys

import numpy as np
import sounddevice as sd
import websockets


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MicClient")

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
CHANNELS = 1
URI = "ws://192.168.4.82:8000/v1/streaming"

FRAMES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))


async def recv_transcripts(websocket):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("message_type")
            text = data.get("text", "")

            if msg_type == "FinalTranscript":
                print(f"\n[FINAL]: {text}")
            elif msg_type == "PartialTranscript":
                # Print partial on same line
                print(f"\r{ text }", end="", flush=True)
    except websockets.exceptions.ConnectionClosed:
        logger.info("\nServer closed the connection.")
    except asyncio.CancelledError:
        # Task cancelled on shutdown
        pass


async def send_mic_audio(websocket):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=20)
    stopped = False

    def audio_callback(indata, frames, time_info, status):
        nonlocal stopped
        if stopped:
            return
        if status:
            print(status, file=sys.stderr)

        data = bytes(indata)

        try:
            loop.call_soon_threadsafe(queue.put_nowait, data)
        except Exception:
            pass
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=FRAMES_PER_CHUNK,
        callback=audio_callback,
    )

    logger.info("Opening microphone stream...")
    with stream:
        logger.info("Microphone streaming started. Press Ctrl+C to stop.")
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                await websocket.send(chunk)
        except websockets.exceptions.ConnectionClosed:
            logger.info("\nConnection closed while sending audio.")
        except asyncio.CancelledError:
            # Graceful shutdown
            pass
        finally:
            stopped = True
            logger.info("Stopping microphone stream...")


async def main():
    stop_event = asyncio.Event()

    def handle_sigint(*_):
        if not stop_event.is_set():
            logger.info("\nCtrl+C received, shutting down...")
            stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        logger.info(f"Connecting to {URI} ...")
        async with websockets.connect(URI) as websocket:
            try:
                handshake = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logger.info(f"Server handshake: {handshake}")
            except asyncio.TimeoutError:
                logger.info("No handshake received from server, continuing anyway.")

            recv_task = asyncio.create_task(recv_transcripts(websocket))
            send_task = asyncio.create_task(send_mic_audio(websocket))

            done, pending = await asyncio.wait(
                {recv_task, send_task, stop_event.wait()},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

    except ConnectionRefusedError:
        logger.error("Cannot connect to server. Is the server running?")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
