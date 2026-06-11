"""Mapping image tokens to panels / regions on the Qwen3-VL token grid.

Qwen3-VL exposes the vision-token grid via `image_grid_thw` (t, h, w) in
patch units; the LM sees the grid after a `spatial_merge_size` (default 2)
merge, so the LM image-token grid is (t, h//merge, w//merge), flattened
row-major. Panels occupy contiguous column ranges proportional to their
pixel widths.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def get_merged_grid_shape(image_grid_thw, spatial_merge: int) -> tuple[int, int, int]:
    t, h, w = image_grid_thw[0].tolist()
    merge = max(1, int(spatial_merge))
    return int(t), int(h) // merge, int(w) // merge


def _monotonic_boundaries(total_columns: int, widths: Sequence[int]) -> list[int]:
    """Split `total_columns` grid columns proportionally to panel pixel widths,
    forcing boundaries to be strictly increasing so every panel gets >=1 column."""
    n_regions = len(widths)
    if n_regions == 0:
        return [0, total_columns]

    total_width = max(1, int(sum(widths)))
    raw = [0.0]
    running = 0
    for width in widths:
        running += int(width)
        raw.append(total_columns * running / total_width)

    boundaries = [0]
    for region_idx in range(1, n_regions):
        proposed = int(round(raw[region_idx]))
        remaining = n_regions - region_idx
        min_allowed = boundaries[-1] + (1 if total_columns >= n_regions else 0)
        max_allowed = total_columns - remaining if total_columns >= n_regions else total_columns
        boundaries.append(min(max(proposed, min_allowed), max_allowed))
    boundaries.append(total_columns)
    return boundaries


def assign_panels_to_tokens(
    image_grid_thw,
    panel_widths: Sequence[int],
    spatial_merge: int,
) -> tuple[np.ndarray, tuple[int, int, int], list[tuple[int, int]]]:
    """Assign each image token to a panel index based on its grid column.

    Returns (panel_ids, grid_shape, panel_column_ranges) where panel_ids has
    one entry per LM image token (row-major over the merged grid).
    """
    t, merged_h, merged_w = get_merged_grid_shape(image_grid_thw, spatial_merge)
    boundaries = _monotonic_boundaries(merged_w, panel_widths)
    panel_ranges = [(boundaries[idx], boundaries[idx + 1]) for idx in range(len(panel_widths))]

    column_to_panel = np.empty((merged_w,), dtype=int)
    for panel_idx, (col_start, col_end) in enumerate(panel_ranges):
        column_to_panel[col_start:col_end] = panel_idx

    panel_ids = np.tile(column_to_panel, t * merged_h)
    return panel_ids, (t, merged_h, merged_w), panel_ranges


def region_positions_from_ids(
    img_start: int,
    region_ids: np.ndarray,
    n_regions: int,
) -> dict[int, list[int]]:
    """Convert per-token region ids into absolute sequence positions per region."""
    positions: dict[int, list[int]] = {}
    for region_idx in range(n_regions):
        indices = np.where(region_ids == region_idx)[0]
        positions[region_idx] = (indices + img_start).tolist()
    return positions


def region_attention_sums(
    attention_values: np.ndarray,
    region_ids: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    return np.bincount(
        region_ids,
        weights=attention_values.astype(np.float64),
        minlength=n_regions,
    )[:n_regions]


def bbox_to_token_positions(
    bbox_xyxy: tuple[float, float, float, float],
    grid_shape: tuple[int, int, int],
    image_size: tuple[int, int],
    img_start: int,
) -> tuple[np.ndarray, list[int]]:
    """Map a pixel-space bounding box to image-token positions.

    A token is selected when its grid-cell center falls inside the box.
    Returns (cell_mask, target_positions): an (mh, mw) boolean mask over grid
    cells and the corresponding absolute token positions in the sequence.
    """
    x0, y0, x1, y1 = bbox_xyxy
    x0, x1 = sorted([float(x0), float(x1)])
    y0, y1 = sorted([float(y0), float(y1)])

    t, mh, mw = grid_shape
    image_w, image_h = image_size

    col_centers = (np.arange(mw) + 0.5) * image_w / mw
    row_centers = (np.arange(mh) + 0.5) * image_h / mh
    cx, cy = np.meshgrid(col_centers, row_centers)
    cell_mask = (x0 <= cx) & (cx <= x1) & (y0 <= cy) & (cy <= y1)

    token_mask = np.tile(cell_mask.reshape(-1), t)
    target_positions = [int(img_start + idx) for idx, keep in enumerate(token_mask.tolist()) if keep]
    return cell_mask, target_positions


def reshape_token_values(
    token_values: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reshape per-token values into the (mh, mw) grid for overlay plotting."""
    t, merged_h, merged_w = grid_shape
    usable = min(token_values.shape[0], t * merged_h * merged_w)
    values = token_values[:usable]
    if usable < t * merged_h * merged_w:
        padded = np.zeros((t * merged_h * merged_w,), dtype=np.float64)
        padded[:usable] = values
        values = padded
    return values.reshape(t, merged_h, merged_w).mean(axis=0)
