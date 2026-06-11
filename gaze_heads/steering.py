"""Attention-mask steering hooks and decode-time attention tracking.

Steering works by registering a forward pre-hook on selected LM self-attention
modules that adds a per-head additive bias to the `attention_mask` kwarg
before softmax: +bias on the target region's image tokens and/or -bias on
everything else, depending on the intervention mode (see common.py). Only the
chosen heads are touched; all other heads in the same layer see the original
mask.

Tracking works by registering forward hooks that record each decode step's
post-softmax attention row, which is later aggregated into per-panel
trajectories (optionally weighted by per-token value norms, following
Kobayashi et al. 2020).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from gaze_heads.common import DEFAULT_INTERVENTION, DEFAULT_SWAP_BIAS, INTERVENTION_MODES
from gaze_heads.regions import region_attention_sums


def _lm_layers(model):
    from gaze_heads.modeling import language_model_layers
    return language_model_layers(model)


@dataclass
class DecodeStepCounter:
    step: int = 0


def group_heads_by_layer(head_specs: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for layer_idx, head_idx in head_specs:
        grouped.setdefault(int(layer_idx), []).append(int(head_idx))
    return grouped


def intervention_positions(
    mode: str,
    target_positions: Sequence[int],
    other_image_positions: Sequence[int],
    img_start: int,
    img_end: int,
    prompt_length: int,
) -> tuple[list[int], list[int], bool]:
    """Compute (suppress_positions, boost_positions, pad_with_suppress) for a mode.

    Modes:
        - "boost_suppress": +bias on target image tokens, -bias on non-target
          image tokens. Text attention untouched. (paper recipe)
        - "suppress_image": -bias on non-target image tokens only.
        - "suppress_all": -bias on non-target image tokens AND non-image
          prompt text. Decoded tokens still natural.
        - "max_suppress_all": same as "suppress_all" plus -bias on every
          decoded token too (grows with kv length).
    """
    target = list(target_positions)
    other = list(other_image_positions)
    if mode == "boost_suppress":
        return other, target, False
    if mode == "suppress_image":
        return other, [], False
    if mode in ("suppress_all", "max_suppress_all"):
        non_image_prompt = list(range(0, img_start)) + list(range(img_end, prompt_length))
        pad_with_suppress = mode == "max_suppress_all"
        return other + non_image_prompt, [], pad_with_suppress
    raise ValueError(
        f"Unknown intervention mode: {mode!r}. Valid modes: {INTERVENTION_MODES}"
    )


def make_static_attention_mask_hook(
    head_indices: Sequence[int],
    suppress_positions: Sequence[int],
    boost_positions: Sequence[int],
    n_query_heads: int,
    device: str,
    swap_bias: float = DEFAULT_SWAP_BIAS,
    decode_only: bool = True,
    pad_with_suppress: bool = False,
):
    """Hook that biases attention toward a fixed target for the whole generation."""
    all_positions = list(suppress_positions) + list(boost_positions)
    if not all_positions or not head_indices:
        def hook(_module, args, kwargs):
            return
        return hook

    base_len = max(all_positions) + 1
    base_bias = torch.zeros((1, n_query_heads, 1, base_len), dtype=torch.float32, device=device)
    h_idx = list(head_indices)
    if suppress_positions:
        s_idx = torch.tensor(list(suppress_positions), dtype=torch.long, device=device)
        for h in h_idx:
            base_bias[0, h, 0, s_idx] = -float(swap_bias)
    if boost_positions:
        b_idx = torch.tensor(list(boost_positions), dtype=torch.long, device=device)
        for h in h_idx:
            base_bias[0, h, 0, b_idx] = float(swap_bias)

    # Caches: dtype conversions of the base bias, and suppression tails for
    # positions past base_len (so we don't reallocate every decode step).
    dtype_cache: dict[torch.dtype, torch.Tensor] = {}
    tail_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    head_tensor = (
        torch.tensor(h_idx, dtype=torch.long, device=device)
        if pad_with_suppress and h_idx
        else None
    )

    def hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            return
        if decode_only and mask.shape[-2] > 1:
            return

        if mask.dtype not in dtype_cache:
            dtype_cache[mask.dtype] = base_bias.to(mask.dtype)
        bias = dtype_cache[mask.dtype]

        kv_len = int(mask.shape[-1])
        if kv_len <= base_len:
            padded = bias[:, :, :, :kv_len]
        else:
            pad_len = kv_len - base_len
            if head_tensor is not None:
                tail_key = (pad_len, mask.dtype)
                if tail_key not in tail_cache:
                    tail = torch.zeros(
                        (1, n_query_heads, 1, pad_len),
                        dtype=mask.dtype,
                        device=device,
                    )
                    tail.index_fill_(1, head_tensor, -float(swap_bias))
                    tail_cache[tail_key] = tail
                padded = torch.cat([bias, tail_cache[tail_key]], dim=-1)
            else:
                padded = torch.nn.functional.pad(bias, (0, pad_len))

        new_kwargs = dict(kwargs)
        new_kwargs["attention_mask"] = mask + padded
        return args, new_kwargs

    return hook


def make_dynamic_attention_mask_hook(
    head_indices: Sequence[int],
    region_positions: dict[int, list[int]],
    schedule: Sequence[tuple[int, int]],
    step_counter: DecodeStepCounter,
    n_query_heads: int,
    device: str,
    swap_bias: float = DEFAULT_SWAP_BIAS,
    decode_only: bool = True,
    intervention: str = DEFAULT_INTERVENTION,
    img_start: int = 0,
    img_end: int = 0,
    prompt_length: int = 0,
):
    """Hook that re-targets the bias mid-generation following `schedule`,
    a sequence of (start_decode_step, target_region) pairs. The shared
    `step_counter` is advanced by a decode-counter hook (see
    register_decode_counter)."""
    if not head_indices:
        def hook(_module, args, kwargs):
            return
        return hook

    h_idx = list(head_indices)
    region_indices = sorted(region_positions.keys())

    # Non-image prompt positions are the same regardless of target region.
    if intervention in ("suppress_all", "max_suppress_all"):
        non_image_prompt_positions = (
            list(range(0, img_start)) + list(range(img_end, prompt_length))
        )
    else:
        non_image_prompt_positions = []
    pad_with_suppress = intervention == "max_suppress_all"

    # base_len must cover every position any target could touch.
    all_touched: list[int] = list(non_image_prompt_positions)
    for positions in region_positions.values():
        all_touched.extend(positions)
    if not all_touched:
        def hook(_module, args, kwargs):
            return
        return hook
    base_len = max(all_touched) + 1

    # Build one bias tensor per target region.
    per_target_bias: dict[int, torch.Tensor] = {}
    for target_region in region_indices:
        target_positions = region_positions[target_region]
        other_image_positions: list[int] = []
        for region_idx in region_indices:
            if region_idx != target_region:
                other_image_positions.extend(region_positions[region_idx])

        suppress_positions, boost_positions, _ = intervention_positions(
            mode=intervention,
            target_positions=target_positions,
            other_image_positions=other_image_positions,
            img_start=img_start,
            img_end=img_end,
            prompt_length=prompt_length,
        )

        bias = torch.zeros((1, n_query_heads, 1, base_len), dtype=torch.float32, device=device)
        if suppress_positions:
            s_idx = torch.tensor(list(suppress_positions), dtype=torch.long, device=device)
            for h in h_idx:
                bias[0, h, 0, s_idx] = -float(swap_bias)
        if boost_positions:
            b_idx = torch.tensor(list(boost_positions), dtype=torch.long, device=device)
            for h in h_idx:
                bias[0, h, 0, b_idx] = float(swap_bias)
        per_target_bias[target_region] = bias

    dtype_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    tail_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    head_tensor = (
        torch.tensor(h_idx, dtype=torch.long, device=device)
        if pad_with_suppress
        else None
    )

    def current_target(step: int) -> int:
        target_region = int(schedule[0][1])
        for start_step, region_idx in schedule:
            if step >= int(start_step):
                target_region = int(region_idx)
            else:
                break
        return target_region

    def hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            return
        if decode_only and mask.shape[-2] > 1:
            return

        target_region = current_target(step_counter.step)
        key = (target_region, mask.dtype)
        if key not in dtype_cache:
            dtype_cache[key] = per_target_bias[target_region].to(mask.dtype)
        bias = dtype_cache[key]

        kv_len = int(mask.shape[-1])
        if kv_len <= base_len:
            padded = bias[:, :, :, :kv_len]
        else:
            pad_len = kv_len - base_len
            if head_tensor is not None:
                tail_key = (pad_len, mask.dtype)
                if tail_key not in tail_cache:
                    tail = torch.zeros(
                        (1, n_query_heads, 1, pad_len),
                        dtype=mask.dtype,
                        device=device,
                    )
                    tail.index_fill_(1, head_tensor, -float(swap_bias))
                    tail_cache[tail_key] = tail
                padded = torch.cat([bias, tail_cache[tail_key]], dim=-1)
            else:
                padded = torch.nn.functional.pad(bias, (0, pad_len))

        new_kwargs = dict(kwargs)
        new_kwargs["attention_mask"] = mask + padded
        return args, new_kwargs

    return hook


def make_decode_step_counter_hook(step_counter: DecodeStepCounter):
    def hook(_module, _inputs, output):
        attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if attn_weights is not None and attn_weights.shape[2] == 1:
            step_counter.step += 1
        return output

    return hook


def register_attention_trackers(model, layers: Sequence[int], records: list[tuple[int, np.ndarray]]):
    """Record each decode step's attention row (n_heads, kv_len) per layer."""
    handles = []
    for layer_idx in layers:
        def make_hook(current_layer: int):
            def hook(_module, _inputs, output):
                attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if attn_weights is not None and attn_weights.shape[2] == 1:
                    records.append(
                        (
                            current_layer,
                            attn_weights[0, :, 0, :].detach().float().cpu().numpy(),
                        )
                    )
                return output

            return hook

        handle = _lm_layers(model)[int(layer_idx)].self_attn.register_forward_hook(
            make_hook(int(layer_idx))
        )
        handles.append(handle)
    return handles


def register_prefill_value_norm_trackers(
    model,
    layers: Sequence[int],
    img_start: int,
    img_end: int,
    storage: dict[int, np.ndarray],
):
    """Capture per-head value-vector norms of the image tokens at prefill,
    used to value-weight raw attention (Kobayashi et al. 2020). GQA value
    heads are repeated to match the query-head count."""
    handles = []
    for layer_idx in layers:
        def make_hook(current_layer: int):
            def hook(module, args, kwargs):
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None:
                    if len(args) == 0:
                        return
                    hidden_states = args[0]
                if hidden_states.shape[1] <= 1:
                    return

                value_states = module.v_proj(hidden_states)
                bsz, seq_len, packed_dim = value_states.shape
                head_dim = int(
                    getattr(
                        module,
                        "head_dim",
                        getattr(getattr(module, "config", None), "head_dim", 0),
                    )
                )
                if head_dim <= 0:
                    raise ValueError("Could not infer head_dim for value-weighted attention.")

                num_query_heads = getattr(module, "num_heads", None)
                if num_query_heads is None:
                    num_query_heads = getattr(getattr(module, "config", None), "num_attention_heads", None)
                if num_query_heads is None and hasattr(module, "q_proj"):
                    num_query_heads = int(module.q_proj.out_features // head_dim)
                if num_query_heads is None:
                    raise ValueError("Could not infer query head count for value-weighted attention.")
                num_query_heads = int(num_query_heads)

                num_key_value_heads = getattr(module, "num_key_value_heads", None)
                if num_key_value_heads is None:
                    num_key_value_heads = getattr(getattr(module, "config", None), "num_key_value_heads", None)
                if num_key_value_heads is None:
                    num_key_value_heads = int(packed_dim // head_dim)
                num_key_value_heads = int(num_key_value_heads)

                value_states = value_states.view(bsz, seq_len, num_key_value_heads, head_dim)
                value_states = value_states.permute(0, 2, 1, 3)
                value_norms = torch.linalg.vector_norm(value_states[0], dim=-1)

                if value_norms.shape[0] != num_query_heads:
                    groups = max(1, num_query_heads // max(1, value_norms.shape[0]))
                    value_norms = value_norms.repeat_interleave(groups, dim=0)
                    value_norms = value_norms[:num_query_heads]

                storage[current_layer] = (
                    value_norms[:, img_start:img_end].detach().float().cpu().numpy()
                )
                return

            return hook

        handle = _lm_layers(model)[int(layer_idx)].self_attn.register_forward_pre_hook(
            make_hook(int(layer_idx)),
            with_kwargs=True,
        )
        handles.append(handle)
    return handles


def register_mask_hooks(model, hook_by_layer: dict[int, object]):
    handles = []
    for layer_idx, hook_fn in hook_by_layer.items():
        handle = _lm_layers(model)[int(layer_idx)].self_attn.register_forward_pre_hook(
            hook_fn,
            with_kwargs=True,
        )
        handles.append(handle)
    return handles


def register_decode_counter(model, layer_idx: int, step_counter: DecodeStepCounter):
    return _lm_layers(model)[int(layer_idx)].self_attn.register_forward_hook(
        make_decode_step_counter_hook(step_counter)
    )


def remove_handles(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def compute_region_trajectory(
    records: Sequence[tuple[int, np.ndarray]],
    heads_by_layer: dict[int, list[int]],
    img_start: int,
    img_end: int,
    region_ids: np.ndarray,
    n_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate tracked raw attention into (normalized, raw) per-step
    per-region trajectories of shape (n_steps, n_regions)."""
    if not heads_by_layer:
        return np.zeros((0, n_regions)), np.zeros((0, n_regions))

    layers = sorted(heads_by_layer.keys())
    n_layers = len(layers)
    n_steps = len(records) // n_layers
    traj_raw = np.zeros((n_steps, n_regions), dtype=np.float64)
    traj_norm = np.zeros((n_steps, n_regions), dtype=np.float64)
    usable = min(max(0, int(img_end - img_start)), int(region_ids.shape[0]))
    region_ids = region_ids[:usable]

    for step_idx in range(n_steps):
        region_scores = np.zeros((n_regions,), dtype=np.float64)
        for offset, layer_idx in enumerate(layers):
            record_idx = step_idx * n_layers + offset
            if record_idx >= len(records):
                break
            stored_layer, attn_values = records[record_idx]
            if int(stored_layer) != int(layer_idx):
                continue
            for head_idx in heads_by_layer[layer_idx]:
                if int(head_idx) >= attn_values.shape[0]:
                    continue
                image_attention = attn_values[int(head_idx), img_start : img_start + usable]
                region_scores += region_attention_sums(image_attention, region_ids, n_regions)

        traj_raw[step_idx] = region_scores
        denom = float(region_scores.sum())
        if denom > 0:
            traj_norm[step_idx] = region_scores / denom

    return traj_norm, traj_raw


def aggregate_step_token_attention_to_regions(
    step_token_attention: np.ndarray,
    region_ids: np.ndarray,
    n_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    if step_token_attention.size == 0:
        return np.zeros((0, n_regions), dtype=np.float64), np.zeros((0, n_regions), dtype=np.float64)

    usable = min(step_token_attention.shape[1], int(region_ids.shape[0]))
    region_ids = region_ids[:usable]
    step_token_attention = step_token_attention[:, :usable]

    n_steps = step_token_attention.shape[0]
    traj_raw = np.zeros((n_steps, n_regions), dtype=np.float64)
    traj_norm = np.zeros((n_steps, n_regions), dtype=np.float64)

    for step_idx in range(n_steps):
        region_scores = region_attention_sums(step_token_attention[step_idx], region_ids, n_regions)
        traj_raw[step_idx] = region_scores
        denom = float(region_scores.sum())
        if denom > 0:
            traj_norm[step_idx] = region_scores / denom

    return traj_norm, traj_raw


def compute_mean_token_attention(
    records: Sequence[tuple[int, np.ndarray]],
    heads_by_layer: dict[int, list[int]],
    img_start: int,
    img_end: int,
) -> np.ndarray:
    """Mean raw attention per image token over all tracked heads and steps."""
    n_image_tokens = max(0, int(img_end - img_start))
    token_sum = np.zeros((n_image_tokens,), dtype=np.float64)
    count = 0

    for layer_idx, attn_values in records:
        if int(layer_idx) not in heads_by_layer:
            continue
        for head_idx in heads_by_layer[int(layer_idx)]:
            if int(head_idx) >= attn_values.shape[0]:
                continue
            token_sum += attn_values[int(head_idx), img_start:img_end]
            count += 1

    if count == 0:
        return token_sum
    return token_sum / float(count)


def compute_step_value_weighted_token_attention(
    records: Sequence[tuple[int, np.ndarray]],
    heads_by_layer: dict[int, list[int]],
    img_start: int,
    img_end: int,
    value_norms_by_layer: dict[int, np.ndarray],
) -> np.ndarray:
    """Per-step per-image-token attention, weighting each token's attention
    by its prefill value norm. Shape (n_steps, n_image_tokens)."""
    if not heads_by_layer:
        return np.zeros((0, max(0, int(img_end - img_start))), dtype=np.float64)

    layers = sorted(heads_by_layer.keys())
    n_layers = len(layers)
    n_steps = len(records) // n_layers
    n_image_tokens = max(0, int(img_end - img_start))
    step_attn = np.zeros((n_steps, n_image_tokens), dtype=np.float64)
    step_counts = np.zeros((n_steps,), dtype=np.float64)

    for step_idx in range(n_steps):
        for offset, layer_idx in enumerate(layers):
            record_idx = step_idx * n_layers + offset
            if record_idx >= len(records):
                break
            stored_layer, attn_values = records[record_idx]
            if int(stored_layer) != int(layer_idx):
                continue
            if int(layer_idx) not in value_norms_by_layer:
                continue
            layer_value_norms = value_norms_by_layer[int(layer_idx)]
            for head_idx in heads_by_layer[layer_idx]:
                if int(head_idx) >= attn_values.shape[0]:
                    continue
                if int(head_idx) >= layer_value_norms.shape[0]:
                    continue
                token_slice = attn_values[int(head_idx), img_start:img_end]
                value_slice = layer_value_norms[int(head_idx)]
                usable = min(token_slice.shape[0], n_image_tokens, value_slice.shape[0])
                step_attn[step_idx, :usable] += token_slice[:usable] * value_slice[:usable]
                step_counts[step_idx] += 1.0

    for step_idx in range(n_steps):
        if step_counts[step_idx] > 0:
            step_attn[step_idx] /= float(step_counts[step_idx])

    return step_attn
