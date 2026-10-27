from __future__ import annotations

import time
from typing import Tuple

import torch
from torch import nn

from .dataset import CLASS_NAMES, LayoutSample, sample_to_payload
from .hooks import ActivationCapture, serialize_activation_hits, serialize_activations


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    capture: ActivationCapture,
    sample: LayoutSample,
    device: torch.device,
) -> Tuple[dict, float]:
    capture.clear()
    model.eval()
    x = torch.tensor(sample.layout / 9.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    start = time.perf_counter()
    logits = model(x)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probabilities = torch.softmax(logits.detach().cpu()[0], dim=0)
    confidence = {CLASS_NAMES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASS_NAMES))}
    predicted = int(probabilities.argmax())

    payload = {
        "type": "inference",
        "sample": sample_to_payload(sample),
        "prediction": {
            "class_id": predicted,
            "class_name": CLASS_NAMES[predicted],
            "confidence": confidence,
        },
        "activations": serialize_activations(capture.activations),
        "activation_hits": serialize_activation_hits(capture.activations),
        "inference_ms": round(elapsed_ms, 3),
    }
    return payload, elapsed_ms
