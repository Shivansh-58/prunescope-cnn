from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Set

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .dataset import CLASS_NAMES, ELEMENTS, generate_synthetic_samples, sample_to_payload
from .hooks import ActivationCapture
from .inference import run_inference
from .model import LayoutCNN
from .train import load_model, save_model, train_model


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "layout_cnn.pt"

app = FastAPI(title="PruneScope CNN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
samples = generate_synthetic_samples(size=6000)
model = LayoutCNN(num_classes=len(CLASS_NAMES)).to(device)
capture = ActivationCapture(model)
capture.register()

clients: Set[WebSocket] = set()
selected_index = 0
playing = False
training_task: asyncio.Task | None = None
training_stop_event = asyncio.Event()
last_frame_time = time.perf_counter()


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/meta")
async def meta():
    return {
        "classes": CLASS_NAMES,
        "elements": ELEMENTS,
        "sample_count": len(samples),
        "device": str(device),
    }


@app.get("/api/sample/{index}")
async def get_sample(index: int):
    return sample_to_payload(samples[index % len(samples)])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json({"type": "meta", "classes": CLASS_NAMES, "elements": ELEMENTS, "sample_count": len(samples), "device": str(device)})
    await send_current_inference(websocket)

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            await handle_command(message, websocket)
    except WebSocketDisconnect:
        clients.discard(websocket)


async def handle_command(message: dict, websocket: WebSocket) -> None:
    global selected_index, playing

    command = message.get("command")
    if command == "next":
        selected_index = (selected_index + 1) % len(samples)
        await broadcast_inference()
    elif command == "prev":
        selected_index = (selected_index - 1) % len(samples)
        await broadcast_inference()
    elif command == "random":
        selected_index = random.randrange(len(samples))
        await broadcast_inference()
    elif command == "step":
        selected_index = (selected_index + 1) % len(samples)
        await broadcast_inference()
    elif command == "play":
        playing = True
    elif command == "pause":
        playing = False
    elif command == "replay":
        await send_current_inference(websocket)
    elif command == "start_training":
        await start_training()
    elif command == "stop_training":
        training_stop_event.set()
    elif command == "save_model":
        save_model(model, MODEL_PATH)
        await broadcast({"type": "model", "status": f"Saved {MODEL_PATH.name}"})
    elif command == "load_model":
        error = load_model(model, MODEL_PATH, device)
        await broadcast({"type": "model", "status": error or f"Loaded {MODEL_PATH.name}"})
        await broadcast_inference()


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(playback_loop())


async def playback_loop() -> None:
    global selected_index
    while True:
        if playing and clients:
            selected_index = (selected_index + 1) % len(samples)
            await broadcast_inference()
            await asyncio.sleep(1 / 24)
        else:
            await asyncio.sleep(0.05)


async def send_current_inference(websocket: WebSocket) -> None:
    payload = build_inference_payload()
    await websocket.send_json(payload)


async def broadcast_inference() -> None:
    await broadcast(build_inference_payload())


def build_inference_payload() -> dict:
    global last_frame_time
    payload, _elapsed = run_inference(model, capture, samples[selected_index], device)
    now = time.perf_counter()
    fps = 1.0 / max(now - last_frame_time, 1e-6)
    last_frame_time = now
    payload["fps"] = round(fps, 1)
    return payload


async def broadcast(payload: dict) -> None:
    stale = []
    for websocket in clients:
        try:
            await websocket.send_json(payload)
        except RuntimeError:
            stale.append(websocket)
    for websocket in stale:
        clients.discard(websocket)


async def start_training() -> None:
    global training_task
    if training_task and not training_task.done():
        await broadcast({"type": "training", "training": True})
        return
    training_stop_event.clear()
    training_task = asyncio.create_task(
        train_model(
            model=model,
            samples=samples,
            device=device,
            status_callback=broadcast,
            stop_event=training_stop_event,
        )
    )
