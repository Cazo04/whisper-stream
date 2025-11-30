import asyncio
import json
import logging
import sys

import websockets

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Client")

CHUNK_DURATION_MS = 200
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000)) * 2 

async def stream_audio(audio_file_path):
    uri = "ws://localhost:8000/v1/streaming"
    logger.info(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            response = await websocket.recv()
            logger.info(f"Server Handshake: {response}")

            logger.info(f"Streaming {audio_file_path}...")
            
            with open(audio_file_path, "rb") as f:
                f.read(44)
                
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    try:
                        await websocket.send(chunk)
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("Error: Connection closed by server abruptly during send.")
                        return

                    await asyncio.sleep(CHUNK_DURATION_MS / 1000)

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        data = json.loads(response)
                        
                        if data.get('message_type') == 'FinalTranscript':
                            print(f"\n[FINAL]: {data['text']}")
                        elif data.get('message_type') == 'PartialTranscript':
                            print(f"\r: {data['text']}", end="", flush=True)
                            
                    except asyncio.TimeoutError:
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("\nServer closed connection.")
                        return

            logger.info("\nFinished sending audio file.")
            logger.info("Waiting for final responses...")
            
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    if data.get('message_type') == 'FinalTranscript':
                        print(f"\n[FINAL]: {data['text']}")
            except asyncio.TimeoutError:
                logger.info("No more responses.")
            except websockets.exceptions.ConnectionClosed:
                pass

    except ConnectionRefusedError:
        logger.error("Cannot connect to server. Is the server running?")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_test.py <path_to_16k_wav_file>")
        sys.exit(1)
    
    asyncio.run(stream_audio(sys.argv[1]))