"""Plot styling and shared figure helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

# Headless backend for scripts; leave the backend alone inside Jupyter so the
# interactive notebook keeps inline rendering.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaze_heads.common import ensure_dir


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 18,
        }
    )


def save_figure_pdf(fig: plt.Figure, path: Path | str, pad_inches: float = 0.1) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    fig.savefig(out, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return out


def draw_attention_overlay(
    ax,
    image,
    heatmap: np.ndarray,
    title: str,
    region_boundaries: Optional[Sequence[tuple[int, int]]] = None,
    cmap: str = "jet",
    alpha: float = 0.75,
) -> None:
    """Overlay a token-grid heatmap on an image (used by the notebook)."""
    width, height = image.size
    ax.imshow(image)
    overlay = ax.imshow(
        heatmap,
        cmap=cmap,
        alpha=alpha,
        extent=(0, width, height, 0),
        interpolation="bilinear",
    )
    ax.set_title(title)
    ax.axis("off")

    if region_boundaries is not None and len(region_boundaries) > 1:
        n_cols = heatmap.shape[1]
        for col_start, _ in region_boundaries[1:]:
            x_pos = width * (col_start / max(n_cols, 1))
            ax.axvline(x=x_pos, color="white", linestyle="--", linewidth=1.5, alpha=0.8)

    ax._gaze_overlay = overlay  # type: ignore[attr-defined]


def add_overlay_colorbar(fig: plt.Figure, ax) -> None:
    overlay = getattr(ax, "_gaze_overlay", None)
    if overlay is not None:
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04, label="Attention")
