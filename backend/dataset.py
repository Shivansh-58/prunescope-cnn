from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["Blog", "Dashboard", "Login", "Ecommerce", "Landing"]

ELEMENTS: Dict[int, str] = {
    0: "Background",
    1: "Header",
    2: "Navbar",
    3: "Button",
    4: "Image",
    5: "Text",
    6: "Footer",
    7: "Sidebar",
    8: "Form",
    9: "Card",
}


@dataclass(frozen=True)
class LayoutSample:
    index: int
    label: int
    label_name: str
    layout: np.ndarray


def _rect(canvas: np.ndarray, x: int, y: int, w: int, h: int, value: int) -> None:
    canvas[y : y + h, x : x + w] = value


def _jitter(value: int, amount: int = 1, low: int = 1, high: int = 30) -> int:
    return max(low, min(high, value + random.randint(-amount, amount)))


def _add_noise(canvas: np.ndarray, chance: float = 0.018) -> None:
    mask = np.random.random(canvas.shape) < chance
    if mask.any():
        canvas[mask] = np.random.randint(0, 10, size=int(mask.sum()))


def generate_layout(label: int) -> np.ndarray:
    canvas = np.zeros((32, 32), dtype=np.float32)

    if label == 0:
        _rect(canvas, 0, 0, 32, _jitter(5), 1)
        _rect(canvas, 2, 7, 28, 3, 4)
        for y in range(12, 25, 4):
            _rect(canvas, 3, y, random.randint(17, 26), 2, 5)
        _rect(canvas, 0, 28, 32, 4, 6)
    elif label == 1:
        _rect(canvas, 0, 0, 32, 4, 1)
        _rect(canvas, 0, 4, 6, 28, 7)
        for y in (6, 16):
            for x in (8, 20):
                _rect(canvas, x, y, 9, 8, 9)
        _rect(canvas, 24, 1, 6, 2, 3)
    elif label == 2:
        _rect(canvas, 0, 0, 32, 5, 1)
        _rect(canvas, 9, 10, 14, 13, 8)
        _rect(canvas, 11, 13, 10, 2, 5)
        _rect(canvas, 11, 17, 10, 2, 5)
        _rect(canvas, 13, 21, 6, 2, 3)
        _rect(canvas, 0, 28, 32, 4, 6)
    elif label == 3:
        _rect(canvas, 0, 0, 32, 4, 1)
        _rect(canvas, 0, 4, 32, 3, 2)
        for y in (9, 20):
            for x in (3, 12, 21):
                _rect(canvas, x, y, 7, 8, 9)
                _rect(canvas, x + 1, y + 1, 5, 4, 4)
                _rect(canvas, x + 2, y + 6, 3, 1, 3)
    else:
        _rect(canvas, 0, 0, 32, 5, 1)
        _rect(canvas, 2, 7, 28, 9, 4)
        _rect(canvas, 5, 18, 16, 2, 5)
        _rect(canvas, 5, 22, 9, 2, 3)
        _rect(canvas, 0, 28, 32, 4, 6)

    _add_noise(canvas)
    return canvas


def generate_synthetic_samples(size: int = 6000, seed: int = 7) -> List[LayoutSample]:
    random.seed(seed)
    np.random.seed(seed)
    samples: List[LayoutSample] = []
    for index in range(size):
        label = index % len(CLASS_NAMES)
        layout = generate_layout(label)
        samples.append(LayoutSample(index, label, CLASS_NAMES[label], layout))
    random.shuffle(samples)
    return [LayoutSample(index, s.label, s.label_name, s.layout) for index, s in enumerate(samples)]


class SyntheticLayoutDataset(Dataset):
    def __init__(self, samples: List[LayoutSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        x = torch.tensor(sample.layout / 9.0, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y


def sample_to_payload(sample: LayoutSample) -> dict:
    return {
        "index": sample.index,
        "label": sample.label,
        "label_name": sample.label_name,
        "layout": sample.layout.astype(int).tolist(),
    }
