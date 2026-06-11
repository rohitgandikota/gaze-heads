"""Shared constants, paths, and small IO helpers."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Root folder of six-panel comics (comicN/p1.png..p6.png subdirectories).
# Override with the GAZE_COMICS_ROOT environment variable or the --comics-root
# CLI flag on each script.
DEFAULT_COMICS_ROOT = Path(os.environ.get("GAZE_COMICS_ROOT", str(REPO_ROOT / "data" / "comics")))

DEFAULT_SEED = 42
DEFAULT_N_PANELS = 6
DEFAULT_TARGET_HEIGHT = 256
DEFAULT_GAP = 6

# Additive pre-softmax attention bias used by the steering hooks.
# 10000.0 saturates to +/-inf in bfloat16, which is the paper recipe.
DEFAULT_SWAP_BIAS = 10000.0

# How the attention-mask intervention treats positions outside the target:
#   - "boost_suppress"    : +bias on target image tokens, -bias on other image
#                           tokens (paper recipe).
#   - "suppress_image"    : -bias on non-target image tokens only.
#   - "suppress_all"      : -bias on non-target image tokens and prompt text.
#   - "max_suppress_all"  : "suppress_all" plus -bias on decoded tokens.
INTERVENTION_MODES = ("boost_suppress", "suppress_image", "suppress_all", "max_suppress_all")
DEFAULT_INTERVENTION = "boost_suppress"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class OutputPaths:
    experiment_name: str
    figures_dir: Path
    logs_dir: Path


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_output_paths(experiment_name: str) -> OutputPaths:
    figures_dir = ensure_dir(REPO_ROOT / "figures" / experiment_name)
    logs_dir = ensure_dir(REPO_ROOT / "logs" / experiment_name)
    return OutputPaths(
        experiment_name=experiment_name,
        figures_dir=figures_dir,
        logs_dir=logs_dir,
    )


def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{suffixes.get(n % 10, 'th')}"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def dump_json(path: Path | str, data: Any) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(data, indent=2, default=json_default))
    return out


def write_text(path: Path | str, text: str) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(text)
    return out
