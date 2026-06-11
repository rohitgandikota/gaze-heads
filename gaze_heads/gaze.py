"""Gaze-score computation and head ranking.

The gaze score of head (l, h) is the mean raw post-softmax attention mass the
final prompt token places on panel k's image tokens when the prompt asks
about panel k, averaged over panels and strips (the diagonal of the 6x6
queried-panel x attended-panel matrix).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gaze_heads.common import dump_json, ordinal
from gaze_heads.modeling import extract_prefill_attentions, find_image_token_range


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def panel_query_prompt(panel_index: int, n_panels: int = 6) -> str:
    count = _NUMBER_WORDS.get(n_panels, str(n_panels))
    return (
        f"Look carefully at this {count}-panel comic strip. "
        f"What is happening in the {ordinal(panel_index)} panel from the left? "
        "Answer briefly."
    )


def collect_last_query_attentions(model: Any, inputs: Any) -> np.ndarray:
    """One prefill pass; return attention from the last prompt token to every
    key position, stacked as (n_layers, n_heads, seq_len)."""
    output = extract_prefill_attentions(model, inputs)
    layers = []
    for attention in output.attentions:
        layers.append(attention[0, :, -1, :].detach().float().cpu().numpy())
    return np.stack(layers, axis=0)


def aggregate_region_attention(
    attn_at_query: np.ndarray,
    inputs: Any,
    processor: Any,
    region_ids: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    """Sum raw attention over each panel's image tokens (no normalization).

    Keeping the attention raw — rather than re-normalizing across panels —
    means a head only scores high if it (a) actually puts attention mass on
    image tokens and (b) concentrates that mass on the queried panel. A head
    that ignores images entirely could still look perfectly diagonal after
    normalization; raw scoring sends it to zero. Rows do NOT sum to 1.
    """
    img_start, img_end = find_image_token_range(inputs, processor)
    image_attention = attn_at_query[:, :, img_start:img_end]
    usable = min(image_attention.shape[-1], region_ids.shape[0])
    image_attention = image_attention[:, :, :usable].astype(np.float64)
    region_ids_usable = region_ids[:usable]

    n_regions = int(n_regions)
    region_onehot = np.zeros((usable, n_regions), dtype=np.float64)
    region_onehot[np.arange(usable), region_ids_usable] = 1.0
    return np.einsum("lht,tr->lhr", image_attention, region_onehot)


def rank_heads_by_score(scores: np.ndarray) -> list[dict[str, float]]:
    ranked = []
    n_layers, n_heads = scores.shape
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ranked.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "score": float(scores[layer_idx, head_idx]),
                }
            )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked


def save_head_ranking(path: Path, ranking: Sequence[dict[str, float]]) -> Path:
    return dump_json(path, list(ranking))


def load_head_ranking(path: Path, top_k: int) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing gaze-head ranking at {path}. Run 01_discover_gaze_heads.py first."
        )
    ranking = json.loads(path.read_text())
    out = []
    for row in ranking[: max(1, int(top_k))]:
        out.append((int(row["layer"]), int(row["head"])))
    if not out:
        raise RuntimeError(f"No heads found in {path}")
    return out


def sample_non_gaze_heads(
    n_layers: int,
    n_heads: int,
    exclude: set[tuple[int, int]],
    n_select: int,
    seed: int,
    scores: np.ndarray | None = None,
    max_score: float | None = None,
) -> list[tuple[int, int]]:
    """Seeded random control heads, optionally restricted to heads whose gaze
    score is below `max_score` (e.g. the bottom-5% percentile cutoff)."""
    candidates = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            if (layer_idx, head_idx) in exclude:
                continue
            if scores is not None and max_score is not None:
                if float(scores[layer_idx, head_idx]) > max_score:
                    continue
            candidates.append((layer_idx, head_idx))
    if not candidates:
        return []

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(candidates), generator=generator).tolist()
    return [candidates[idx] for idx in perm[: min(n_select, len(candidates))]]
