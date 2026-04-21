from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def resolve_torch_device(preferred: str | None = "auto") -> torch.device:
    value = str(preferred or "auto").strip().lower()
    if value in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def infer_hf_device(preferred: str | int | None = "auto") -> int:
    if isinstance(preferred, int):
        return preferred if preferred >= 0 and torch.cuda.is_available() else -1
    value = str(preferred or "auto").strip().lower()
    if value in {"auto", "cuda"}:
        return 0 if torch.cuda.is_available() else -1
    if value == "cpu":
        return -1
    try:
        parsed = int(value)
    except ValueError:
        return 0 if torch.cuda.is_available() else -1
    return parsed if parsed >= 0 and torch.cuda.is_available() else -1


def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=device.type == "cuda")
    if isinstance(batch, Mapping):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def dataloader_device_kwargs(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"pin_memory": False}
    return {
        "pin_memory": True,
        "persistent_workers": True,
    }
