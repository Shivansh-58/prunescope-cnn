from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class ActivationCapture:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.handles = []

    def register(self) -> None:
        self.close()
        for name in ("conv1", "conv2", "conv3", "conv4"):
            module = getattr(self.model, name)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.activations[name] = output.detach().cpu()

        return hook

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _feature_grid(tensor: torch.Tensor, max_channels: int = 16) -> List[List[List[float]]]:
    feature = tensor[0, :max_channels]
    min_value = float(feature.min())
    max_value = float(feature.max())
    denom = max(max_value - min_value, 1e-6)
    normalized = (feature - min_value) / denom
    return normalized.numpy().round(4).tolist()


def serialize_activations(activations: Dict[str, torch.Tensor]) -> Dict[str, list]:
    return {name: _feature_grid(tensor) for name, tensor in activations.items()}


def serialize_activation_hits(activations: Dict[str, torch.Tensor]) -> Dict[str, list]:
    hits = {}
    for name, tensor in activations.items():
        feature = tensor[0]
        flat = feature.flatten(1)
        thresholds = torch.quantile(flat, 0.8, dim=1)
        active_mask = feature > thresholds[:, None, None]
        hit_frequency = active_mask.float().mean(dim=(1, 2))
        means = feature.mean(dim=(1, 2))
        peak_scale = feature.amax(dim=(1, 2)) / max(float(feature.max()), 1e-6)
        score = hit_frequency * 0.7 + peak_scale * 0.3
        min_value = float(score.min())
        max_value = float(score.max())
        normalized = (score - min_value) / max(max_value - min_value, 1e-6)
        hits[name] = normalized.numpy().round(4).tolist()
    return hits
