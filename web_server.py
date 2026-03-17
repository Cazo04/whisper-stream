import asyncio
import json
import logging
import os
import re
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WebServer")


class AccessNoiseFilter(logging.Filter):
    """Filter noisy scanner traffic from uvicorn access logs."""

    _scan_path_pattern = re.compile(
        r"^/(?:\.git|\.env|\.DS_Store|\.vscode|graphql|api(?:/|$)|swagger|v2/api-docs|v3/api-docs|"
        r"actuator|server-status|console|login\.action|trace\.axd|info\.php|webjars|@vite|"
        r"\.well-known|ecp|s/|config\.json|telescope|about)",
        re.IGNORECASE,
    )

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True

        method = str(args[1]).upper()
        path = str(args[2])
        status_code = str(args[4])

        if status_code in {"404", "405"} and method in {"GET", "POST", "HEAD", "OPTIONS"}:
            if self._scan_path_pattern.match(path):
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(AccessNoiseFilter())

app = FastAPI(title="Audio Transcription Web Server")

# Mount static files
app.mount("/src", StaticFiles(directory="src"), name="src")

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()

# Store active ESP connection for broadcasting to browser
esp_connection: WebSocket | None = None

# Configuration
BACKEND_STREAMING_SERVER = "ws://localhost:8000/v1/streaming"
SAMPLE_RATE = 16000
CHANNELS = 1


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=500
        )


@app.head("/")
async def head_index():
    """Return 200 for HEAD checks to avoid 405 noise."""
    return HTMLResponse(content="", status_code=200)


@app.options("/")
async def options_index():
    """Return 200 for OPTIONS checks to avoid 405 noise."""
    return HTMLResponse(content="", status_code=200)


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for browser clients
    Receives audio from browser, forwards to backend server, and sends back transcripts
    """
    global esp_connection
    
    # Check if a client is already connected (but allow if ESP is connected for display only)
    if len(active_connections) >= 1:
        # Still allow connection, but check if there's an active audio session
        logger.info(f"New browser client connecting (active: {len(active_connections)})")
    
    await websocket.accept()
    active_connections.add(websocket)
    
    client_id = id(websocket)
    logger.info(f"Browser Client {client_id} connected")
    
    backend_ws = None
    recv_task = None
    send_task = None
    
    try:
        # Connect to backend streaming server
        import websockets
        logger.info(f"Browser {client_id}: Connecting to backend {BACKEND_STREAMING_SERVER}")
        backend_ws = await websockets.connect(BACKEND_STREAMING_SERVER)
        # Send identification frame to backend.
        await backend_ws.send(json.dumps({"client_type": "browser"}))
        
        # Wait for handshake from backend
        try:
            handshake = await asyncio.wait_for(backend_ws.recv(), timeout=5.0)
            logger.info(f"Browser {client_id}: Backend handshake received: {handshake}")
            # Send handshake to client
            await websocket.send_json({
                "message_type": "ServerReady",
                "text": "Connected to transcription server"
            })
        except asyncio.TimeoutError:
            logger.warning(f"Browser {client_id}: No handshake from backend, continuing anyway")
            await websocket.send_json({
                "message_type": "ServerReady",
                "text": "Connected (no handshake)"
            })
        
        # Task to receive transcripts from backend and forward to client
        async def receive_from_backend():
            try:
                async for message in backend_ws:
                    try:
                        # Forward transcript to browser client
                        data = json.loads(message)
                        await websocket.send_json(data)
                        
                        # Log transcripts
                        msg_type = data.get("message_type")
                        text = data.get("text", "")
                        if msg_type == "FinalTranscript":
                            logger.info(f"Browser {client_id} [FINAL]: {text}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Browser {client_id}: JSON decode error: {e}")
                        continue
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Browser {client_id}: Backend connection closed")
            except Exception as e:
                logger.error(f"Browser {client_id}: Error receiving from backend: {e}")
        
        # Task to receive audio from client and forward to backend
        async def send_to_backend():
            try:
                while True:
                    # Receive audio data from browser
                    data = await websocket.receive()
                    
                    if "bytes" in data:
                        # Forward audio bytes to backend
                        audio_bytes = data["bytes"]
                        await backend_ws.send(audio_bytes)
                    elif "text" in data:
                        # Handle control messages from client
                        try:
                            msg = json.loads(data["text"])
                            msg_type = msg.get("type")
                            
                            if msg_type == "stop":
                                logger.info(f"Browser {client_id}: Stop requested")
                                break
                            elif msg_type == "ping":
                                # Forward ping to backend so latency reflects end-to-end path.
                                await backend_ws.send(data["text"])
                            elif msg_type in ["config", "esp_start", "esp_stop", "reset_session"]:
                                # Forward runtime control/config to backend
                                logger.info(f"Browser {client_id}: Config update: {msg}")
                                await backend_ws.send(data["text"])
                        except json.JSONDecodeError:
                            pass
                    
            except WebSocketDisconnect:
                logger.info(f"Browser {client_id}: Disconnected")
            except RuntimeError as e:
                logger.info(f"Browser {client_id}: Connection already closed: {e}")
            except Exception as e:
                logger.error(f"Browser {client_id}: Error sending to backend: {e}")
        
        # Run both tasks concurrently
        recv_task = asyncio.create_task(receive_from_backend())
        send_task = asyncio.create_task(send_to_backend())
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            {recv_task, send_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except ConnectionRefusedError:
        logger.error(f"Browser {client_id}: Cannot connect to backend server")
        try:
            await websocket.send_json({
                "message_type": "Error",
                "text": "Cannot connect to transcription backend"
            })
        except:
            pass
    except Exception as e:
        logger.error(f"Browser {client_id}: Unexpected error: {e}")
        try:
            await websocket.send_json({
                "message_type": "Error",
                "text": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        # Cleanup
        active_connections.discard(websocket)
        
        if backend_ws:
            try:
                await backend_ws.close()
            except:
                pass
        
        if recv_task and not recv_task.done():
            recv_task.cancel()
        if send_task and not send_task.done():
            send_task.cancel()
            
        logger.info(f"Browser {client_id}: Connection closed and cleaned up")


@app.websocket("/wsesp")
async def websocket_esp_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ESP32 clients
    ESP connects to this endpoint for:
    1. Sending audio to server
    2. Receiving DisplayText for OLED display
    3. Forwarding transcripts to browser clients
    """
    global esp_connection
    
    await websocket.accept()
    esp_connection = websocket
    
    client_id = id(websocket)
    logger.info(f"ESP Client {client_id} connected")
    
    backend_ws = None
    recv_task = None
    send_task = None
    
    try:
        # Connect to backend streaming server
        import websockets
        logger.info(f"ESP {client_id}: Connecting to backend {BACKEND_STREAMING_SERVER}")
        backend_ws = await websockets.connect(BACKEND_STREAMING_SERVER)
        # Send identification frame to backend.
        await backend_ws.send(json.dumps({"client_type": "esp"}))
        
        # Wait for handshake from backend
        try:
            handshake = await asyncio.wait_for(backend_ws.recv(), timeout=5.0)
            logger.info(f"ESP {client_id}: Backend handshake received")
            # Forward handshake to ESP
            await websocket.send_json(json.loads(handshake))
        except asyncio.TimeoutError:
            logger.warning(f"ESP {client_id}: No handshake from backend")
        
        # Task to receive messages from backend and forward to ESP + broadcast to browsers
        async def receive_from_backend():
            try:
                async for message in backend_ws:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("message_type")

                        # Keep ESP payload lean: only push control/config/display events to device.
                        forward_to_esp = msg_type in {
                            "SessionBegins", "ConfigAck", "DisplayText", "RuntimeConfig",
                            "esp_start", "esp_stop", "DisplayError", "Error", "Pong",
                        }
                        if forward_to_esp:
                            await websocket.send_json(data)
                        
                        # If this is a transcript message, broadcast to all browser clients
                        if msg_type in ["FinalTranscript", "TranslatedTranscript", "SpeechDetected", "SilenceDetected"]:
                            # Broadcast to all active browser connections
                            disconnected = set()
                            for browser_ws in active_connections:
                                try:
                                    await browser_ws.send_json(data)
                                except:
                                    disconnected.add(browser_ws)
                            
                            # Clean up disconnected browsers
                            for ws in disconnected:
                                active_connections.discard(ws)
                            
                            if msg_type == "FinalTranscript":
                                logger.info(f"ESP {client_id} [BROADCAST]: {data.get('text', '')}")
                        
                        # Log DisplayText for ESP
                        if msg_type == "DisplayText":
                            logger.info(f"ESP {client_id} [DISPLAY]: {data.get('text', '')}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"ESP {client_id}: JSON decode error: {e}")
                        continue
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"ESP {client_id}: Backend connection closed")
            except Exception as e:
                logger.error(f"ESP {client_id}: Error receiving from backend: {e}")
        
        # Task to receive messages from ESP and forward to backend
        async def send_to_backend():
            try:
                while True:
                    data = await websocket.receive()
                    
                    if "text" in data:
                        try:
                            msg = json.loads(data["text"])
                            msg_type = msg.get("type")
                            
                            # Add client_type to identify as ESP
                            if msg_type in ["config", "esp_start", "esp_stop"]:
                                if "client_type" not in msg:
                                    msg["client_type"] = "esp"
                                await backend_ws.send(json.dumps(msg))
                            else:
                                # Forward other messages as-is
                                await backend_ws.send(data["text"])
                                
                            logger.info(f"ESP {client_id}: Forwarded {msg_type}")
                        except json.JSONDecodeError:
                            pass
                    elif "bytes" in data:
                        # Forward binary audio data
                        await backend_ws.send(data["bytes"])
                    
            except WebSocketDisconnect:
                logger.info(f"ESP {client_id}: Disconnected")
            except RuntimeError as e:
                logger.info(f"ESP {client_id}: Connection already closed: {e}")
            except Exception as e:
                logger.error(f"ESP {client_id}: Error sending to backend: {e}")
        
        # Run both tasks concurrently
        recv_task = asyncio.create_task(receive_from_backend())
        send_task = asyncio.create_task(send_to_backend())
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            {recv_task, send_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except ConnectionRefusedError:
        logger.error(f"ESP {client_id}: Cannot connect to backend server")
        try:
            await websocket.send_json({
                "message_type": "Error",
                "text": "Cannot connect to transcription backend"
            })
        except:
            pass
    except Exception as e:
        logger.error(f"ESP {client_id}: Unexpected error: {e}")
        try:
            await websocket.send_json({
                "message_type": "Error",
                "text": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        esp_connection = None
        
        if backend_ws:
            try:
                await backend_ws.close()
            except:
                pass
        
        if recv_task and not recv_task.done():
            recv_task.cancel()
        if send_task and not send_task.done():
            send_task.cancel()
            
        logger.info(f"ESP {client_id}: Connection closed and cleaned up")


@app.get("/api/status")
async def get_status():
    """Health check endpoint"""
    return {
        "status": "online",
        "active_connections": len(active_connections),
        "esp_connected": esp_connection is not None,
        "backend_server": BACKEND_STREAMING_SERVER,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS
    }


if __name__ == "__main__":
    logger.info("Starting Audio Transcription Web Server...")
    logger.info(f"Backend streaming server: {BACKEND_STREAMING_SERVER}")
    cert_file = os.getenv("SSL_CERTFILE")
    key_file = os.getenv("SSL_KEYFILE")

    if cert_file and key_file:
        logger.info("Web UI will be available at: https://localhost:8080")
    else:
        logger.warning("TLS cert/key not set, running without HTTPS/WSS")
        logger.info("Web UI will be available at: http://localhost:8080")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
    )
