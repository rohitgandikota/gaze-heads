#!/usr/bin/env python3
"""Track gaze-head attention trajectories during free panel-by-panel narration.

The model narrates the strip with no steering while we record decode-time
attention for (a) the top-K gaze heads and (b) K random non-gaze control
heads. Gaze heads produce a staircase that climbs panel by panel in sync with
the generated text; control heads show no panel structure.

Outputs per comic (under figures/narration_trajectory/<comic>/):
  - gaze_heads_trajectory.pdf
  - non_gaze_heads_trajectory.pdf
plus raw trajectories (.npy), generated texts, and a summary.json in logs/.

Usage:
    python 02_narration_trajectory.py --comics-root /path/to/comics --max-comics 5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaze_heads.common import (
    DEFAULT_COMICS_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_N_PANELS,
    DEFAULT_SEED,
    dump_json,
    make_output_paths,
    write_text,
)
from gaze_heads.data import build_strip, list_comic_dirs
from gaze_heads.gaze import load_head_ranking, sample_non_gaze_heads
from gaze_heads.modeling import (
    decode_generated_text,
    decode_generated_tokens,
    find_image_token_range,
    load_model_and_processor,
    model_dims,
    prepare_inputs,
    run_generation,
)
from gaze_heads.plots import save_figure_pdf, set_plot_style
from gaze_heads.regions import assign_panels_to_tokens
from gaze_heads.steering import (
    aggregate_step_token_attention_to_regions,
    compute_region_trajectory,
    compute_step_value_weighted_token_attention,
    group_heads_by_layer,
    register_attention_trackers,
    register_prefill_value_norm_trackers,
    remove_handles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track gaze-head trajectories during narration.")
    parser.add_argument("--comics-root", type=str, default=str(DEFAULT_COMICS_ROOT))
    parser.add_argument("--gaze-ranking", type=str, default="",
                        help="Path to gaze_head_ranking.json (default: logs/gaze_discovery/).")
    parser.add_argument("--comic-name", type=str, default="",
                        help="Run a single comic directory by name.")
    parser.add_argument("--max-comics", type=int, default=0)
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--control-heads", type=int, default=100)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--attention-mode",
        type=str,
        default="value_weighted",
        choices=["value_weighted", "raw"],
        help="value_weighted multiplies attention by per-token value norms (paper figure).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gap", type=int, default=6)
    return parser.parse_args()


def narration_prompt() -> str:
    return (
        "Please describe what happens in each panel, in order. "
        "Start each panel description with 'Panel 1:', 'Panel 2:', and so on."
    )


def find_panel_transitions(
    text: str,
    tokens: list[str],
    n_panels: int,
) -> list[tuple[int, int]]:
    """Locate the decode step at which each 'Panel k:' marker is generated,
    so the trajectory plot can mark where the narration moves on."""
    if not text or not tokens:
        return []

    cumulative_chars = []
    running = 0
    for token in tokens:
        cumulative_chars.append(running)
        running += len(token)

    patterns = [
        r"\*\*Panel\s*([1-9]\d*)",
        r"\bPanel\s*([1-9]\d*)\b",
        r"\bP\s*([1-9]\d*)\b",
    ]
    found: list[tuple[int, int]] = []
    seen_steps: set[int] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            panel_number = int(match.group(1))
            if panel_number < 1 or panel_number > n_panels:
                continue
            char_pos = match.start()
            token_idx = 0
            for idx, start_char in enumerate(cumulative_chars):
                if start_char <= char_pos:
                    token_idx = idx
                else:
                    break
            if token_idx in seen_steps:
                continue
            seen_steps.add(token_idx)
            found.append((token_idx, panel_number))
    found.sort(key=lambda item: item[0])
    return found


def tracked_generation(
    model,
    processor,
    inputs,
    heads_by_layer: dict[int, list[int]],
    img_start: int,
    img_end: int,
    region_ids: np.ndarray,
    n_regions: int,
    max_new_tokens: int,
    attention_mode: str,
):
    records: list[tuple[int, np.ndarray]] = []
    value_norms_by_layer: dict[int, np.ndarray] = {}
    tracker_handles = register_attention_trackers(model, sorted(heads_by_layer.keys()), records)
    value_handles = register_prefill_value_norm_trackers(
        model=model,
        layers=sorted(heads_by_layer.keys()),
        img_start=img_start,
        img_end=img_end,
        storage=value_norms_by_layer,
    )
    try:
        sequences = run_generation(model=model, inputs=inputs, max_new_tokens=max_new_tokens)
    finally:
        remove_handles(tracker_handles)
        remove_handles(value_handles)

    text = decode_generated_text(processor, sequences, int(inputs["input_ids"].shape[1]))
    tokens = decode_generated_tokens(processor, sequences, int(inputs["input_ids"].shape[1]))
    if attention_mode == "value_weighted":
        step_token_attention = compute_step_value_weighted_token_attention(
            records=records,
            heads_by_layer=heads_by_layer,
            img_start=img_start,
            img_end=img_end,
            value_norms_by_layer=value_norms_by_layer,
        )
        traj_norm, traj_raw = aggregate_step_token_attention_to_regions(
            step_token_attention=step_token_attention,
            region_ids=region_ids,
            n_regions=n_regions,
        )
    else:
        traj_norm, traj_raw = compute_region_trajectory(
            records=records,
            heads_by_layer=heads_by_layer,
            img_start=img_start,
            img_end=img_end,
            region_ids=region_ids,
            n_regions=n_regions,
        )
    return text, tokens, traj_norm, traj_raw


def build_trajectory_figure(
    trajectory_raw: np.ndarray,
    transitions: list[tuple[int, int]],
    attention_mode: str,
    title: str,
    shared_vmax: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 5.5))
    panel_labels = [f"P{i+1}" for i in range(DEFAULT_N_PANELS)]
    cbar_label = (
        "Summed value-weighted scores"
        if attention_mode == "value_weighted"
        else "Summed raw attention scores"
    )

    if trajectory_raw.size == 0:
        ax.text(0.5, 0.5, "No decode attention captured.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    ax.grid(False)
    im = ax.imshow(
        trajectory_raw.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="hot",
        vmin=0.0,
        vmax=shared_vmax,
    )
    ax.set_yticks(np.arange(DEFAULT_N_PANELS), labels=panel_labels)
    ax.set_xlabel("Generated Text Tokens")
    ax.set_ylabel("Panel Tokens")
    ax.set_title(title, pad=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)

    for step_idx, panel_number in transitions:
        if step_idx < trajectory_raw.shape[0]:
            ax.axvline(step_idx, color="white", linestyle=":", linewidth=2.2, alpha=0.9)
            ax.text(
                step_idx + 1,
                DEFAULT_N_PANELS - 0.45,
                f"P{panel_number}",
                ha="left",
                va="top",
                fontsize=18,
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def main() -> None:
    args = parse_args()
    set_plot_style()
    outputs = make_output_paths("narration_trajectory")

    comics_root = Path(args.comics_root)
    comic_dirs = list_comic_dirs(comics_root, n_panels=DEFAULT_N_PANELS)
    if args.comic_name:
        comic_dirs = [path for path in comic_dirs if path.name == args.comic_name]
    if args.max_comics > 0:
        comic_dirs = comic_dirs[: args.max_comics]
    if not comic_dirs:
        raise FileNotFoundError(f"No valid 6-panel comic directories found under {comics_root}.")

    ranking_path = (
        Path(args.gaze_ranking)
        if args.gaze_ranking
        else outputs.logs_dir.parent / "gaze_discovery" / "gaze_head_ranking.json"
    )

    model, processor = load_model_and_processor(model_id=args.model_id, device=args.device)
    n_layers, n_heads, spatial_merge = model_dims(model)

    gaze_scores = np.load(ranking_path.parent / "gaze_scores.npy")
    gaze_heads = load_head_ranking(ranking_path, top_k=args.top_k_gaze)
    gaze_by_layer = group_heads_by_layer(gaze_heads)
    # Control: random heads with gaze score below the median.
    median_score = float(np.median(gaze_scores))
    control_heads = sample_non_gaze_heads(
        n_layers=n_layers,
        n_heads=n_heads,
        exclude=set(gaze_heads),
        n_select=args.control_heads,
        seed=args.seed,
        scores=gaze_scores,
        max_score=median_score,
    )
    control_by_layer = group_heads_by_layer(control_heads)

    for comic_dir in comic_dirs:
        strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
        prompt = narration_prompt()
        inputs = prepare_inputs(processor, strip.strip, prompt, args.device)
        img_start, img_end = find_image_token_range(inputs, processor)
        region_ids, _, _ = assign_panels_to_tokens(
            image_grid_thw=inputs["image_grid_thw"],
            panel_widths=strip.panel_widths,
            spatial_merge=spatial_merge,
        )

        gaze_text, gaze_tokens, _, gaze_traj_raw = tracked_generation(
            model=model,
            processor=processor,
            inputs=inputs,
            heads_by_layer=gaze_by_layer,
            img_start=img_start,
            img_end=img_end,
            region_ids=region_ids,
            n_regions=DEFAULT_N_PANELS,
            max_new_tokens=args.max_new_tokens,
            attention_mode=args.attention_mode,
        )
        control_text, _, _, control_traj_raw = tracked_generation(
            model=model,
            processor=processor,
            inputs=inputs,
            heads_by_layer=control_by_layer,
            img_start=img_start,
            img_end=img_end,
            region_ids=region_ids,
            n_regions=DEFAULT_N_PANELS,
            max_new_tokens=args.max_new_tokens,
            attention_mode=args.attention_mode,
        )

        comic_log_dir = outputs.logs_dir / strip.name
        comic_fig_dir = outputs.figures_dir / strip.name
        comic_log_dir.mkdir(parents=True, exist_ok=True)
        comic_fig_dir.mkdir(parents=True, exist_ok=True)

        transitions = find_panel_transitions(
            text=gaze_text,
            tokens=gaze_tokens,
            n_panels=DEFAULT_N_PANELS,
        )
        np.save(comic_log_dir / "gaze_traj_raw.npy", gaze_traj_raw)
        np.save(comic_log_dir / "control_traj_raw.npy", control_traj_raw)
        write_text(comic_log_dir / "gaze_text.txt", gaze_text)
        write_text(comic_log_dir / "control_text.txt", control_text)

        dump_json(
            comic_log_dir / "summary.json",
            {
                "comic_name": strip.name,
                "gaze_head_count": len(gaze_heads),
                "control_head_count": len(control_heads),
                "gaze_text": gaze_text,
                "control_text": control_text,
                "panel_widths": strip.panel_widths,
                "gaze_panel_transitions": transitions,
                "attention_mode": args.attention_mode,
            },
        )

        shared_vmax = max(
            float(gaze_traj_raw.max()) if gaze_traj_raw.size > 0 else 0.0,
            float(control_traj_raw.max()) if control_traj_raw.size > 0 else 0.0,
            1e-6,
        )
        save_figure_pdf(
            build_trajectory_figure(
                trajectory_raw=gaze_traj_raw,
                transitions=transitions,
                attention_mode=args.attention_mode,
                title=f"Gaze Heads (n={len(gaze_heads)})",
                shared_vmax=shared_vmax,
            ),
            comic_fig_dir / "gaze_heads_trajectory.pdf",
        )
        save_figure_pdf(
            build_trajectory_figure(
                trajectory_raw=control_traj_raw,
                transitions=transitions,
                attention_mode=args.attention_mode,
                title=f"Random Non-Gaze Heads (n={len(control_heads)})",
                shared_vmax=shared_vmax,
            ),
            comic_fig_dir / "non_gaze_heads_trajectory.pdf",
        )

        print(f"Saved trajectory outputs for {strip.name}")


if __name__ == "__main__":
    main()
