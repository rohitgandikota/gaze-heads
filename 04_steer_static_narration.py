#!/usr/bin/env python3
"""Causal static narration steering: an ambiguous "this panel" prompt, resolved by the gaze heads.

The model is asked "What is happening in this panel of the comic strip?"
without saying which panel. Unsteered, it answers with the first panel or a
summary of the whole strip. With the gaze heads steered to a target panel,
the answer should describe that panel instead.

For every (strip, target_panel) pair we generate a short (~100 token)
description with the heads held on the target, and a forced 1-of-6 Claude
judge picks which panel the text best matches. HIT iff it matches the
target; outputs essentially identical to the baseline count as misses
(steering that changes nothing is not a hit). Chance = 1/6.

Outputs (under logs/<output-name>/):
  - aggregate_results.json   overall + per-panel accuracy with bootstrap CIs
  - generations.jsonl        every generation with its judgment

Usage:
    python 04_steer_static_narration.py --comics-root /path/to/comics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
    find_image_token_range,
    load_model_and_processor,
    model_dims,
    prepare_inputs,
    run_generation,
)
from gaze_heads.regions import assign_panels_to_tokens, region_positions_from_ids
from gaze_heads.steering import (
    group_heads_by_layer,
    intervention_positions,
    make_static_attention_mask_hook,
    register_mask_hooks,
    remove_handles,
)

PROMPT = "What is happening in this panel of the comic strip?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judged static narration steering (ambiguous 'this panel' prompt).")
    p.add_argument("--comics-root", type=str, default=str(DEFAULT_COMICS_ROOT))
    p.add_argument("--gaze-ranking", type=str, default="",
                   help="Path to gaze_head_ranking.json (default: logs/gaze_discovery/).")
    p.add_argument("--max-comics", type=int, default=0)
    p.add_argument("--top-k-gaze", type=int, default=100)
    p.add_argument("--top-k-random", type=int, default=100)
    p.add_argument("--nongaze-percentile", type=float, default=5.0)
    p.add_argument("--include-all-heads", action="store_true",
                   help="Also run the all-heads control (slow; output is junk).")
    p.add_argument("--targets-per-strip", type=int, default=DEFAULT_N_PANELS,
                   help="Target panels evaluated per strip: 6 = every panel "
                        "(default, paper-style), 1 = one random target (quick).")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gap", type=int, default=6)
    # Steering is applied to prefill AND decode by default, matching the VQA
    # recipe; the prompt's "this panel" is resolved during prefill.
    p.add_argument("--decode-only", dest="decode_only", action="store_true")
    p.add_argument("--full-sequence", dest="decode_only", action="store_false")
    p.set_defaults(decode_only=False)
    p.add_argument("--intervention", type=str, default="boost_suppress", choices=INTERVENTION_MODES)
    p.add_argument("--swap-bias", type=float, default=DEFAULT_SWAP_BIAS)
    p.add_argument("--claude-model", type=str, default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--output-name", type=str, default="steer_static_narration")
    return p.parse_args()


def generate_steered(
    model, processor, inputs, heads_by_layer, panel_positions,
    target_panel: int, n_query_heads: int, device: str,
    decode_only: bool, max_new_tokens: int,
    intervention: str, img_start: int, img_end: int, prompt_length: int,
    swap_bias: float,
) -> str:
    target_positions = panel_positions[target_panel]
    other_positions: list[int] = []
    for p in range(DEFAULT_N_PANELS):
        if p != target_panel:
            other_positions.extend(panel_positions[p])

    suppress_positions, boost_positions, pad_with_suppress = intervention_positions(
        mode=intervention,
        target_positions=target_positions,
        other_image_positions=other_positions,
        img_start=img_start,
        img_end=img_end,
        prompt_length=prompt_length,
    )
    hook_by_layer = {
        layer_idx: make_static_attention_mask_hook(
            head_indices=heads,
            suppress_positions=suppress_positions,
            boost_positions=boost_positions,
            n_query_heads=n_query_heads,
            device=device,
            swap_bias=swap_bias,
            decode_only=decode_only,
            pad_with_suppress=pad_with_suppress,
        )
        for layer_idx, heads in heads_by_layer.items()
    }
    handles = register_mask_hooks(model, hook_by_layer)
    try:
        sequences = run_generation(model=model, inputs=inputs, max_new_tokens=max_new_tokens)
    finally:
        remove_handles(handles)
    return decode_generated_text(processor, sequences, int(inputs["input_ids"].shape[1]))


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

    all_outcomes: dict[str, list[bool]] = {c: [] for c in conditions}
    junk_counts: dict[str, int] = {c: 0 for c in conditions}
    per_panel_outcomes: dict[str, dict[int, list[bool]]] = {
        c: {p: [] for p in range(DEFAULT_N_PANELS)} for c in conditions
    }

    generations_jsonl_path = outputs.logs_dir / "generations.jsonl"
    gen_f = open(generations_jsonl_path, "w")

    rng = np.random.RandomState(args.seed)
    targets_per_strip = max(1, min(int(args.targets_per_strip), DEFAULT_N_PANELS))

    for i, comic_dir in enumerate(comic_dirs):
        strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
        print(f"\n=== Strip {i + 1}/{len(comic_dirs)}: {strip.name} ===", flush=True)

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

        sequences = run_generation(model=model, inputs=inputs, max_new_tokens=args.max_new_tokens)
        baseline_text = decode_generated_text(processor, sequences, int(inputs["input_ids"].shape[1]))

        strip_log_dir = outputs.logs_dir / strip.name
        strip_log_dir.mkdir(parents=True, exist_ok=True)
        write_text(strip_log_dir / "baseline_text.txt", baseline_text)
        print(f"  baseline: {baseline_text[:100]}", flush=True)

        if targets_per_strip >= DEFAULT_N_PANELS:
            targets_for_strip = list(range(DEFAULT_N_PANELS))
        else:
            targets_for_strip = rng.choice(DEFAULT_N_PANELS, size=targets_per_strip, replace=False).tolist()

        for cond_name, heads_by_layer in conditions.items():
            for target_panel in targets_for_strip:
                text = generate_steered(
                    model=model, processor=processor, inputs=inputs,
                    heads_by_layer=heads_by_layer, panel_positions=panel_positions,
                    target_panel=target_panel, n_query_heads=n_query_heads,
                    device=args.device, decode_only=args.decode_only,
                    max_new_tokens=args.max_new_tokens,
                    intervention=args.intervention,
                    img_start=img_start, img_end=img_end,
                    prompt_length=int(inputs["input_ids"].shape[1]),
                    swap_bias=args.swap_bias,
                )

                try:
                    judgment = judge_match_target_panel(
                        strip_image=strip.strip,
                        segment_text=text,
                        baseline_text=baseline_text,
                        target_panel=target_panel + 1,
                        n_panels=DEFAULT_N_PANELS,
                        model_name=args.claude_model,
                    )
                except Exception as exc:
                    judgment = {"matched_panel": None, "is_junk": True, "correct": False, "reasoning": f"error: {exc}"}

                outcome = bool(judgment["correct"])
                is_junk = bool(judgment.get("is_junk", False))
                all_outcomes[cond_name].append(outcome)
                per_panel_outcomes[cond_name][target_panel].append(outcome)
                if is_junk:
                    junk_counts[cond_name] += 1
                matched = judgment.get("matched_panel")
                tag = "HIT" if outcome else (
                    "JUNK" if is_junk else f"MISS->P{matched}" if matched else "MISS"
                )
                print(f"  [{cond_name}] P{target_panel + 1}: {tag} | {text.strip()[:90]}", flush=True)

                gen_f.write(json.dumps({
                    "strip_name": strip.name,
                    "comic_dir": str(comic_dir),
                    "condition": cond_name,
                    "target_panel": target_panel + 1,
                    "baseline_text": baseline_text,
                    "steered_text": text,
                    "judgment": judgment,
                    "experiment": args.output_name,
                }) + "\n")
                gen_f.flush()
                write_text(
                    strip_log_dir / f"{cond_name}_target_{target_panel + 1}_text.txt",
                    text,
                )

    gen_f.close()

    print("\n" + "=" * 80, flush=True)
    print("AGGREGATE (static narration, forced 1-of-6 judge)", flush=True)
    print("=" * 80, flush=True)
    aggregate = {
        "n_strips": len(comic_dirs),
        "targets_per_strip": targets_per_strip,
        "prompt": PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "intervention": args.intervention,
        "decode_only": args.decode_only,
        "chance": 1.0 / DEFAULT_N_PANELS,
        "conditions": {},
    }
    for cond_name in conditions:
        ci = bootstrap_ci(all_outcomes[cond_name], n_bootstrap=10000, ci=0.95, seed=args.seed)
        per_panel = {}
        for panel_idx in range(DEFAULT_N_PANELS):
            ci_p = bootstrap_ci(per_panel_outcomes[cond_name][panel_idx], n_bootstrap=10000, ci=0.95, seed=args.seed)
            per_panel[f"panel_{panel_idx + 1}"] = ci_p
        aggregate["conditions"][cond_name] = {
            "overall": {
                "accuracy": ci["accuracy"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n": ci["n"],
                "junk_count": junk_counts[cond_name],
            },
            "per_panel": per_panel,
        }
        print(
            f"  {cond_name}: acc={ci['accuracy']:.3f} "
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
            f"(n={ci['n']}, junk={junk_counts[cond_name]})",
            flush=True,
        )

    dump_json(outputs.logs_dir / "aggregate_results.json", aggregate)
    print(f"\nSaved {outputs.logs_dir / 'aggregate_results.json'}", flush=True)


if __name__ == "__main__":
    main()
