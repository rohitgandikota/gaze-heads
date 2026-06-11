#!/usr/bin/env python3
"""Discover gaze heads from six-panel comic strips.

For each strip we ask one question per panel ("What is happening in the k-th
panel from the left?") and record the raw post-softmax attention the final
prompt token places on each panel's image tokens. Averaged over strips, every
head gets a 6x6 queried-panel x attended-panel matrix; the gaze score is its
diagonal mean. Heads are ranked by this score and the ranking is consumed by
every downstream steering script.

Outputs (under logs/<output-name>/ and figures/<output-name>/):
  - gaze_head_ranking.json   ranked (layer, head, score) list
  - gaze_scores.npy          (n_layers, n_heads) score matrix
  - mean_panel_attention.npy (n_panels, n_layers, n_heads, n_panels)
  - gaze_score_histogram.pdf, gaze_score_map.pdf
  - gaze_heads.pdf, non_gaze_heads.pdf (example 6x6 matrices)

Usage:
    # panel-folder comics (comicN/p1.png..p6.png)
    python 01_discover_gaze_heads.py --comics-root /path/to/comics --n-samples 500

    # raw COMICS corpus (<comic_id>/<page>_<panel>.jpg): strips are sampled
    # as random consecutive 6-panel windows of interior pages (paper protocol)
    python 01_discover_gaze_heads.py --use-raw --comics-root /path/to/raw_panel_images --n-samples 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from gaze_heads.common import (
    DEFAULT_COMICS_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_N_PANELS,
    DEFAULT_SEED,
    dump_json,
    make_output_paths,
    seed_everything,
)
from gaze_heads.data import build_random_raw_strip, build_strip, list_comic_dirs
from gaze_heads.gaze import (
    aggregate_region_attention,
    collect_last_query_attentions,
    panel_query_prompt,
    rank_heads_by_score,
    save_head_ranking,
)
from gaze_heads.modeling import (
    load_model_and_processor,
    model_dims,
    prepare_inputs,
)
from gaze_heads.plots import save_figure_pdf, set_plot_style
from gaze_heads.regions import assign_panels_to_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover gaze heads from six-panel comic strips.")
    parser.add_argument("--comics-root", type=str, default=str(DEFAULT_COMICS_ROOT),
                        help="Folder of comicN/p1.png..p6.png subdirectories, or with "
                             "--use-raw the root of the raw COMICS corpus.")
    parser.add_argument("--use-raw", action="store_true",
                        help="Treat --comics-root as the raw COMICS corpus "
                             "(<comic_id>/<page>_<panel>.jpg) and sample strips as "
                             "random consecutive panel windows (seeded, reproducible).")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of comics to use (capped at what exists).")
    parser.add_argument("--n-panels", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k-preview", type=int, default=6,
                        help="How many example head matrices to plot.")
    parser.add_argument("--output-name", type=str, default="gaze_discovery")
    return parser.parse_args()


def plot_score_histogram(
    scores: np.ndarray,
    top10_threshold: float | None = None,
    top100_threshold: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    flat_scores = scores.reshape(-1)
    ax.hist(flat_scores, bins=60, color="tab:blue", alpha=0.88, edgecolor="white")
    if top100_threshold is not None:
        ax.axvline(top100_threshold, color="tab:orange", linestyle=":", linewidth=1.5,
                   label=f"Top 100 ({top100_threshold:.3f})")
    if top10_threshold is not None:
        ax.axvline(top10_threshold, color="tab:red", linestyle=":", linewidth=1.5,
                   label=f"Top 10 ({top10_threshold:.3f})")
    ax.set_xlabel("Gaze score (raw image-token attention, diagonal mean)")
    ax.set_ylabel("Number of heads")
    ax.set_title("Gaze score distribution")
    ax.legend()
    return fig


def plot_score_map(scores: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    im = ax.imshow(scores, aspect="auto", cmap="viridis")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Gaze score by layer and head")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Score")
    return fig


def plot_head_heatmaps_grid(
    mean_panel_attention: np.ndarray,
    selected_heads: list[dict[str, float]],
    suptitle: str,
) -> plt.Figure:
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.2, 7.6))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax in axes.flat:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0)

    for plot_idx, (ax, row) in enumerate(zip(axes.flat, selected_heads)):
        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        layer_idx = int(row["layer"])
        head_idx = int(row["head"])
        score = float(row["score"])
        matrix = mean_panel_attention[:, layer_idx, head_idx, :]
        ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(float(matrix.max()), 1e-6))
        ax.set_title(f"L{layer_idx} H{head_idx} ({score:.3f})")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))

        if row_idx == n_rows - 1:
            ax.set_xticklabels([f"P{i+1}" for i in range(matrix.shape[1])])
            ax.set_xlabel("Attended Panel")
            ax.tick_params(axis="x", labelbottom=True, pad=2)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False, bottom=False)

        if col_idx == 0:
            ax.set_yticklabels([f"P{i+1}" for i in range(matrix.shape[0])])
            ax.set_ylabel("Queried Panel")
            ax.tick_params(axis="y", labelleft=True, left=True, length=3, pad=2)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, left=False)

    for ax in axes.flat[len(selected_heads):]:
        ax.axis("off")

    fig.suptitle(suptitle, y=0.97)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.09, top=0.90, wspace=0.04, hspace=0.15)
    return fig


def select_random_non_gaze_heads(
    ranked_heads: list[dict[str, float]],
    n_select: int,
    seed: int,
) -> list[dict[str, float]]:
    """Pick low-scoring example heads from the bottom half of the ranking."""
    pool = ranked_heads[len(ranked_heads) // 2 :]
    if not pool:
        pool = ranked_heads
    rng = np.random.RandomState(seed)
    chosen_idx = rng.choice(len(pool), size=min(n_select, len(pool)), replace=False)
    selected = [pool[int(idx)] for idx in chosen_idx]
    selected.sort(key=lambda row: (int(row["layer"]), int(row["head"])))
    return selected


def main() -> None:
    args = parse_args()
    set_plot_style()
    seed_everything(args.seed)
    outputs = make_output_paths(args.output_name)

    comics_root = Path(args.comics_root)
    if args.use_raw:
        rng = np.random.RandomState(args.seed)

        def strip_source():
            for _ in range(args.n_samples):
                try:
                    yield build_random_raw_strip(
                        raw_panel_root=comics_root,
                        rng=rng,
                        n_panels=args.n_panels,
                        target_height=args.target_height,
                        gap=args.gap,
                    )
                except (RuntimeError, FileNotFoundError) as exc:
                    print(f"Raw strip sampling failed: {exc}")
                    return

        print(f"Sampling {args.n_samples} raw strips from {comics_root}", flush=True)
    else:
        comic_dirs = list_comic_dirs(comics_root, n_panels=args.n_panels)
        if not comic_dirs:
            raise FileNotFoundError(
                f"No valid {args.n_panels}-panel comic directories under {comics_root}. "
                "Expected comicN/p1.png..pN.png (see README)."
            )
        comic_dirs = comic_dirs[: args.n_samples]

        def strip_source():
            for comic_dir in comic_dirs:
                yield build_strip(
                    panel_dir=comic_dir,
                    n_panels=args.n_panels,
                    target_height=args.target_height,
                    gap=args.gap,
                )

        print(f"Using {len(comic_dirs)} comics from {comics_root}", flush=True)

    model, processor = load_model_and_processor(model_id=args.model_id, device=args.device)
    n_layers, n_heads, spatial_merge = model_dims(model)

    gaze_sum = np.zeros((args.n_panels, n_layers, n_heads, args.n_panels), dtype=np.float64)
    sampled_names: list[str] = []
    valid_samples = 0

    for strip in tqdm(strip_source(), total=args.n_samples, desc="Gaze discovery"):
        region_ids = None
        ok = True
        per_prompt = []

        for panel_index in range(1, args.n_panels + 1):
            prompt = panel_query_prompt(panel_index, n_panels=args.n_panels)
            try:
                inputs = prepare_inputs(processor, strip.strip, prompt, args.device)
                if region_ids is None:
                    region_ids, _, _ = assign_panels_to_tokens(
                        image_grid_thw=inputs["image_grid_thw"],
                        panel_widths=strip.panel_widths,
                        spatial_merge=spatial_merge,
                    )
                attn_at_query = collect_last_query_attentions(model, inputs)
                panel_attention = aggregate_region_attention(
                    attn_at_query=attn_at_query,
                    inputs=inputs,
                    processor=processor,
                    region_ids=region_ids,
                    n_regions=args.n_panels,
                )
                per_prompt.append(panel_attention)
            except Exception as exc:
                print(f"Skipping sample {strip.name} on panel {panel_index}: {exc}")
                ok = False
                break

        if not ok or len(per_prompt) != args.n_panels:
            continue

        for prompt_idx in range(args.n_panels):
            gaze_sum[prompt_idx] += per_prompt[prompt_idx]
        sampled_names.append(strip.name)
        valid_samples += 1

    if valid_samples == 0:
        raise RuntimeError("No valid samples were processed. Check the comics root.")

    mean_panel_attention = gaze_sum / float(valid_samples)
    gaze_scores = np.zeros((n_layers, n_heads), dtype=np.float32)
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            diag = [mean_panel_attention[p, layer_idx, head_idx, p] for p in range(args.n_panels)]
            gaze_scores[layer_idx, head_idx] = float(np.mean(diag))

    ranked_heads = rank_heads_by_score(gaze_scores)
    top_gaze_heads = ranked_heads[: max(1, args.top_k_preview)]
    random_non_gaze_heads = select_random_non_gaze_heads(
        ranked_heads=ranked_heads,
        n_select=max(1, args.top_k_preview),
        seed=args.seed + 1,
    )

    np.save(outputs.logs_dir / "gaze_scores.npy", gaze_scores)
    np.save(outputs.logs_dir / "mean_panel_attention.npy", mean_panel_attention.astype(np.float32))
    save_head_ranking(outputs.logs_dir / "gaze_head_ranking.json", ranked_heads)

    summary = {
        "config": {
            "comics_root": args.comics_root,
            "use_raw": args.use_raw,
            "model_id": args.model_id,
            "device": args.device,
            "n_samples_requested": args.n_samples,
            "n_panels": args.n_panels,
            "target_height": args.target_height,
            "gap": args.gap,
            "seed": args.seed,
            "score_kind": "raw_image_token_attention_sum",
        },
        "n_valid_samples": valid_samples,
        "sample_names": sampled_names[:50],
        "top_heads": ranked_heads[:50],
    }
    dump_json(outputs.logs_dir / "summary.json", summary)

    top10_threshold = float(ranked_heads[9]["score"]) if len(ranked_heads) > 9 else None
    top100_threshold = float(ranked_heads[99]["score"]) if len(ranked_heads) > 99 else None
    save_figure_pdf(
        plot_score_histogram(gaze_scores, top10_threshold, top100_threshold),
        outputs.figures_dir / "gaze_score_histogram.pdf",
    )
    save_figure_pdf(plot_score_map(gaze_scores), outputs.figures_dir / "gaze_score_map.pdf")
    save_figure_pdf(
        plot_head_heatmaps_grid(mean_panel_attention, top_gaze_heads, "Gaze Heads"),
        outputs.figures_dir / "gaze_heads.pdf",
    )
    save_figure_pdf(
        plot_head_heatmaps_grid(mean_panel_attention, random_non_gaze_heads, "Non-Gaze Heads"),
        outputs.figures_dir / "non_gaze_heads.pdf",
    )

    print(f"Saved outputs to {outputs.figures_dir} and {outputs.logs_dir}")
    print("Top 10 heads:")
    for row in ranked_heads[:10]:
        print(f"  L{row['layer']:02d} H{row['head']:02d} score={row['score']:.4f}")


if __name__ == "__main__":
    main()
