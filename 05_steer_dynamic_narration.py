#!/usr/bin/env python3
"""Causal dynamic narration steering: switch the gaze target mid-generation.

For each strip:
  1. Sample a derangement of the panels (no panel in its default position) as
     the steering schedule.
  2. Generate one ~300-token narration while the gaze-head target switches to
     the next scheduled panel every --switch-every decode steps.
  3. Chunk the narration into segments aligned with the schedule and ask a
     forced 1-of-6 Claude judge which panel each segment describes.
  4. Report per-segment accuracy and the Spearman correlation between the
     steering schedule and the panels the model actually described.

Outputs (under logs/<output-name>/):
  - aggregate_results.json   per-segment accuracy + Spearman rho per condition
  - generations.jsonl        every narration with schedule, segments, judgments

Usage:
    python 05_steer_dynamic_narration.py --comics-root /path/to/comics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from gaze_heads.common import (
    DEFAULT_COMICS_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_N_PANELS,
    DEFAULT_SEED,
    DEFAULT_SWAP_BIAS,
    INTERVENTION_MODES,
    dump_json,
    make_output_paths,
    seed_everything,
    write_text,
)
from gaze_heads.data import build_strip, list_comic_dirs
from gaze_heads.gaze import load_head_ranking, sample_non_gaze_heads
from gaze_heads.judge import DEFAULT_JUDGE_MODEL, bootstrap_ci, judge_match_target_panel, require_api_key
from gaze_heads.modeling import (
    decode_generated_text,
    decode_generated_tokens,
    find_image_token_range,
    load_model_and_processor,
    model_dims,
    prepare_inputs,
    run_generation,
)
from gaze_heads.regions import assign_panels_to_tokens, region_positions_from_ids
from gaze_heads.steering import (
    DecodeStepCounter,
    group_heads_by_layer,
    make_dynamic_attention_mask_hook,
    register_decode_counter,
    register_mask_hooks,
    remove_handles,
)

PROMPT = (
    "Please describe each panel of this comic strip in order as I point you "
    "to them. Keep each description SHORT — one brief sentence per panel, "
    "naming the action and the salient objects. Do not summarise the whole "
    "strip first; just describe panel by panel."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judged dynamic narration steering.")
    p.add_argument("--comics-root", type=str, default=str(DEFAULT_COMICS_ROOT))
    p.add_argument("--gaze-ranking", type=str, default="",
                   help="Path to gaze_head_ranking.json (default: logs/gaze_discovery/).")
    p.add_argument("--max-comics", type=int, default=0)
    p.add_argument("--top-k-gaze", type=int, default=100)
    p.add_argument("--top-k-random", type=int, default=100)
    p.add_argument("--nongaze-percentile", type=float, default=5.0)
    p.add_argument("--include-all-heads", action="store_true")
    p.add_argument("--switch-every", type=int, default=50,
                   help="Decode tokens per steering segment.")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gap", type=int, default=6)
    p.add_argument("--decode-only", dest="decode_only", action="store_true")
    p.add_argument("--full-sequence", dest="decode_only", action="store_false")
    p.set_defaults(decode_only=True)
    p.add_argument("--intervention", type=str, default="boost_suppress", choices=INTERVENTION_MODES)
    p.add_argument("--claude-model", type=str, default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--output-name", type=str, default="steer_dynamic_narration")
    # Hard +/-inf steering under pure greedy decoding can lock the KV cache
    # into repetition loops mid-narration; these knobs let the generator
    # escape without changing the steering itself.
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="HuggingFace generate() repetition_penalty (e.g., 1.3).")
    p.add_argument("--no-repeat-ngram-size", type=int, default=None,
                   help="HuggingFace generate() no_repeat_ngram_size (e.g., 4).")
    p.add_argument("--swap-bias", type=float, default=DEFAULT_SWAP_BIAS,
                   help="Steering strength (smaller values give softer transitions).")
    return p.parse_args()


def sample_derangement(rng: np.random.RandomState, n: int) -> list[int]:
    """Permutation with no element in its original position."""
    attempts = 0
    while True:
        perm = rng.permutation(n).tolist()
        if all(perm[i] != i for i in range(n)):
            return perm
        attempts += 1
        if attempts > 200:
            return [(i + 1) % n for i in range(n)]


def segment_tokens(tokens: list[str], switch_every: int, n_segments: int) -> list[str]:
    segments = []
    for seg_idx in range(n_segments):
        start = seg_idx * switch_every
        end = min(start + switch_every, len(tokens))
        segments.append("".join(tokens[start:end]).strip())
    return segments


def run_one(
    model, processor, inputs, heads_by_layer, panel_positions,
    schedule, n_query_heads, device, decode_only, max_new_tokens,
    intervention, img_start, img_end, prompt_length,
    swap_bias: float,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> tuple[str, list[str]]:
    """One narration with a dynamic gaze schedule. Returns (text, tokens)."""
    step_counter = DecodeStepCounter()
    layers = sorted(heads_by_layer.keys()) or [0]

    hook_by_layer = {
        layer_idx: make_dynamic_attention_mask_hook(
            head_indices=head_indices,
            region_positions=panel_positions,
            schedule=schedule,
            step_counter=step_counter,
            n_query_heads=n_query_heads,
            device=device,
            swap_bias=swap_bias,
            decode_only=decode_only,
            intervention=intervention,
            img_start=img_start,
            img_end=img_end,
            prompt_length=prompt_length,
        )
        for layer_idx, head_indices in heads_by_layer.items()
    }
    mask_handles = register_mask_hooks(model, hook_by_layer)
    counter_handle = register_decode_counter(model, layers[-1], step_counter)
    try:
        sequences = run_generation(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    finally:
        remove_handles(mask_handles)
        counter_handle.remove()

    input_len = int(inputs["input_ids"].shape[1])
    text = decode_generated_text(processor, sequences, input_len)
    tokens = decode_generated_tokens(processor, sequences, input_len)
    return text, tokens


def main() -> None:
    args = parse_args()
    require_api_key()
    seed_everything(args.seed)
    outputs = make_output_paths(args.output_name)

    ranking_path = (
        Path(args.gaze_ranking)
        if args.gaze_ranking
        else outputs.logs_dir.parent / "gaze_discovery" / "gaze_head_ranking.json"
    )

    model, processor = load_model_and_processor(model_id=args.model_id, device=args.device)
    n_layers, n_query_heads, spatial_merge = model_dims(model)

    gaze_scores = np.load(ranking_path.parent / "gaze_scores.npy")
    gaze_top = load_head_ranking(ranking_path, top_k=args.top_k_gaze)
    cutoff = float(np.percentile(gaze_scores, args.nongaze_percentile))
    random_heads = sample_non_gaze_heads(
        n_layers=n_layers, n_heads=n_query_heads, exclude=set(gaze_top),
        n_select=args.top_k_random, seed=args.seed,
        scores=gaze_scores, max_score=cutoff,
    )

    conditions = {
        f"gaze_top{args.top_k_gaze}": group_heads_by_layer(gaze_top),
        f"non_gaze_{args.top_k_random}": group_heads_by_layer(random_heads),
    }
    if args.include_all_heads:
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_query_heads)]
        conditions["all_heads"] = group_heads_by_layer(all_heads)
    print(f"Non-gaze cutoff (raw score percentile {args.nongaze_percentile}) = {cutoff:.4f}", flush=True)

    comics_root = Path(args.comics_root)
    comic_dirs = list_comic_dirs(comics_root, n_panels=DEFAULT_N_PANELS)
    if args.max_comics > 0:
        comic_dirs = comic_dirs[: args.max_comics]
    if not comic_dirs:
        raise FileNotFoundError(f"No valid {DEFAULT_N_PANELS}-panel comic directories under {comics_root}.")

    rng = np.random.RandomState(args.seed)
    all_outcomes: dict[str, list[bool]] = {c: [] for c in conditions}
    spearman_rhos: dict[str, list[float]] = {c: [] for c in conditions}
    junk_counts: dict[str, int] = {c: 0 for c in conditions}
    max_new_tokens = args.switch_every * DEFAULT_N_PANELS

    generations_jsonl_path = outputs.logs_dir / "generations.jsonl"
    gen_f = open(generations_jsonl_path, "w")

    for i, comic_dir in enumerate(comic_dirs):
        strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
        schedule_perm = sample_derangement(rng, DEFAULT_N_PANELS)  # zero-indexed
        target_panels_1idx = [p + 1 for p in schedule_perm]
        print(f"\n=== Strip {i + 1}/{len(comic_dirs)}: {strip.name} | schedule={target_panels_1idx} ===", flush=True)

        inputs = prepare_inputs(processor, strip.strip, PROMPT, args.device)
        img_start, img_end = find_image_token_range(inputs, processor)
        region_ids, _, _ = assign_panels_to_tokens(
            image_grid_thw=inputs["image_grid_thw"],
            panel_widths=strip.panel_widths,
            spatial_merge=spatial_merge,
        )
        panel_positions = region_positions_from_ids(
            img_start=img_start,
            region_ids=region_ids[: max(0, img_end - img_start)],
            n_regions=DEFAULT_N_PANELS,
        )
        schedule = [(j * args.switch_every, panel_idx) for j, panel_idx in enumerate(schedule_perm)]

        strip_log_dir = outputs.logs_dir / strip.name
        strip_log_dir.mkdir(parents=True, exist_ok=True)

        for cond_name, heads_by_layer in conditions.items():
            text, tokens = run_one(
                model=model, processor=processor, inputs=inputs,
                heads_by_layer=heads_by_layer, panel_positions=panel_positions,
                schedule=schedule, n_query_heads=n_query_heads,
                device=args.device, decode_only=args.decode_only,
                max_new_tokens=max_new_tokens, intervention=args.intervention,
                img_start=img_start, img_end=img_end,
                prompt_length=int(inputs["input_ids"].shape[1]),
                swap_bias=args.swap_bias,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            segments = segment_tokens(tokens, args.switch_every, DEFAULT_N_PANELS)
            matched_panels: list[int | None] = []
            per_segment_hits: list[bool] = []
            for seg_idx, (seg_text, target_zero_idx) in enumerate(zip(segments, schedule_perm)):
                target_1idx = target_zero_idx + 1
                if not seg_text.strip():
                    judgment: dict[str, Any] = {"matched_panel": None, "is_junk": True, "correct": False}
                else:
                    try:
                        judgment = judge_match_target_panel(
                            strip_image=strip.strip,
                            segment_text=seg_text,
                            baseline_text=None,
                            target_panel=target_1idx,
                            n_panels=DEFAULT_N_PANELS,
                            model_name=args.claude_model,
                        )
                    except Exception as exc:
                        judgment = {
                            "matched_panel": None, "is_junk": True,
                            "correct": False, "reasoning": f"error: {exc}",
                        }
                matched = judgment.get("matched_panel")
                matched_panels.append(matched)
                hit = bool(judgment.get("correct"))
                per_segment_hits.append(hit)
                if judgment.get("is_junk"):
                    junk_counts[cond_name] += 1
                tag = "HIT" if hit else ("JUNK" if judgment.get("is_junk") else f"MISS->P{matched}")
                print(
                    f"  [{cond_name}] seg {seg_idx + 1}, target=P{target_1idx}: {tag}\n"
                    f"      seg: {seg_text.strip()[:130]}",
                    flush=True,
                )

            all_outcomes[cond_name].extend(per_segment_hits)

            # Spearman rho between the schedule and the matched panels,
            # skipping strips where too many segments were unmatched/junk.
            matched_clean = [m for m in matched_panels if m is not None]
            target_clean = [t for t, m in zip(target_panels_1idx, matched_panels) if m is not None]
            if len(matched_clean) >= 3:
                rho, _ = spearmanr(target_clean, matched_clean)
                if np.isnan(rho):
                    rho = 0.0
                spearman_rhos[cond_name].append(float(rho))

            gen_f.write(json.dumps({
                "strip_name": strip.name,
                "comic_dir": str(comic_dir),
                "condition": cond_name,
                "schedule_zero_indexed": schedule_perm,
                "schedule_one_indexed": target_panels_1idx,
                "switch_every": args.switch_every,
                "full_text": text,
                "segments": segments,
                "matched_panels": matched_panels,
                "per_segment_hits": per_segment_hits,
                "experiment": args.output_name,
            }) + "\n")
            gen_f.flush()
            write_text(strip_log_dir / f"{cond_name}_text.txt", text)

    gen_f.close()

    print("\n" + "=" * 80, flush=True)
    print("AGGREGATE (per-segment LLM judge + Spearman rho per strip)", flush=True)
    print("=" * 80, flush=True)
    aggregate = {
        "n_strips": len(comic_dirs),
        "switch_every": args.switch_every,
        "intervention": args.intervention,
        "decode_only": args.decode_only,
        "chance_per_segment_accuracy": 1.0 / DEFAULT_N_PANELS,
        "conditions": {},
    }
    for cond_name in conditions:
        acc_ci = bootstrap_ci(all_outcomes[cond_name], n_bootstrap=10000, ci=0.95, seed=args.seed)
        rhos = spearman_rhos[cond_name]
        rho_mean = float(np.mean(rhos)) if rhos else 0.0
        rho_std = float(np.std(rhos)) if rhos else 0.0
        aggregate["conditions"][cond_name] = {
            "per_segment_accuracy": acc_ci["accuracy"],
            "ci_low": acc_ci["ci_low"],
            "ci_high": acc_ci["ci_high"],
            "n_segments": acc_ci["n"],
            "spearman_rho_mean": rho_mean,
            "spearman_rho_std": rho_std,
            "n_strips_for_rho": len(rhos),
            "junk_segments": junk_counts[cond_name],
        }
        print(
            f"  {cond_name}: per-seg acc={acc_ci['accuracy']:.3f} "
            f"[{acc_ci['ci_low']:.3f}, {acc_ci['ci_high']:.3f}] | "
            f"Spearman rho={rho_mean:.3f}+/-{rho_std:.3f} (n_strips={len(rhos)}) | "
            f"junk_segs={junk_counts[cond_name]}",
            flush=True,
        )

    dump_json(outputs.logs_dir / "aggregate_results.json", aggregate)
    print(f"\nSaved {outputs.logs_dir / 'aggregate_results.json'}", flush=True)


if __name__ == "__main__":
    main()
