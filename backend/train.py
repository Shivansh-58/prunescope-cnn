from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import SyntheticLayoutDataset


async def train_model(
    model: nn.Module,
    samples,
    device: torch.device,
    status_callback: Callable[[dict], Awaitable[None]],
    stop_event: asyncio.Event,
    epochs: int = 8,
    batch_size: int = 96,
) -> None:
    dataset = SyntheticLayoutDataset(samples)
    train_len = int(len(dataset) * 0.82)
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(1, epochs + 1):
        if stop_event.is_set():
            break

        model.train()
        running_loss = 0.0
        for batch_index, (x, y) in enumerate(train_loader, start=1):
            if stop_event.is_set():
                break

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if batch_index % 8 == 0:
                await status_callback(
                    {
                        "type": "training",
                        "training": True,
                        "epoch": epoch,
                        "loss": round(running_loss / batch_index, 4),
                        "val_accuracy": None,
                    }
                )
            await asyncio.sleep(0)

        val_accuracy = evaluate(model, val_loader, device)
        await status_callback(
            {
                "type": "training",
                "training": not stop_event.is_set(),
                "epoch": epoch,
                "loss": round(running_loss / max(len(train_loader), 1), 4),
                "val_accuracy": round(val_accuracy, 4),
            }
        )

    await status_callback({"type": "training", "training": False})


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predicted = model(x).argmax(dim=1)
        correct += int((predicted == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: Path, device: torch.device) -> Optional[str]:
    if not path.exists():
        return f"No model found at {path}"
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return None
