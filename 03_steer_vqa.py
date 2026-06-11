#!/usr/bin/env python3
"""Causal VQA steering: redirect gaze heads to a target panel and judge the answer.

For every (strip, target_panel) pair:
  1. Generate a baseline VQA answer with no intervention.
  2. Steer the top-K gaze heads' attention onto the target panel
     (boost_suppress: +bias on target image tokens, -bias on every other
     panel's image tokens; text attention untouched).
  3. Ask a Claude judge which panel the steered answer best matches
     (forced 1-of-6; junk counts as a miss; chance = 1/6).

Conditions: gaze top-K, K random non-gaze controls (bottom percentile of the
gaze-score distribution), and optionally all heads (which destroys generation).

Outputs (under logs/<output-name>/):
  - aggregate_results.json   overall + per-panel accuracy with bootstrap CIs
  - generations.jsonl        every generation, judge-agnostic
  - <comic>/...              per-strip texts and judgments

Usage:
    python 03_steer_vqa.py --comics-root /path/to/comics
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

DEFAULT_VQA_PROMPT = (
    "What is the main action or event happening in this comic strip? "
    "Answer briefly."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judged VQA steering over six-panel strips.")
    p.add_argument("--comics-root", type=str, default=str(DEFAULT_COMICS_ROOT))
    p.add_argument("--gaze-ranking", type=str, default="",
                   help="Path to gaze_head_ranking.json (default: logs/gaze_discovery/).")
    p.add_argument("--comic-name", type=str, default="")
    p.add_argument("--start-comic-idx", type=int, default=0,
                   help="Skip the first N comics (for sharding across machines).")
    p.add_argument("--max-comics", type=int, default=0)
    p.add_argument("--top-k-gaze", type=int, default=100)
    p.add_argument("--top-k-random", type=int, default=100)
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=15)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--gap", type=int, default=6)
    p.add_argument("--claude-model", type=str, default=DEFAULT_JUDGE_MODEL)
    # Steering is applied to prefill AND decode by default (the paper recipe);
    # --decode-only restricts it to decode steps.
    p.add_argument("--decode-only", dest="decode_only", action="store_true")
    p.add_argument("--full-sequence", dest="decode_only", action="store_false")
    p.set_defaults(decode_only=False)
    p.add_argument("--intervention", type=str, default="boost_suppress", choices=INTERVENTION_MODES)
    p.add_argument("--no-all-heads", dest="include_all_heads", action="store_false",
                   help="Skip the all-heads control (~2x faster).")
    p.set_defaults(include_all_heads=True)
    p.add_argument("--nongaze-percentile", type=float, default=5.0,
                   help="Non-gaze controls are sampled from heads whose gaze score "
                        "is in the bottom X%% of the distribution. With a looser "
                        "cutoff the strong bias leaks image signal through "
                        "moderately-attending heads and inflates the control.")
    p.add_argument("--swap-bias", type=float, default=DEFAULT_SWAP_BIAS,
                   help="Additive pre-softmax bias magnitude (10000 = +/-inf in bf16).")
    p.add_argument("--prompt", type=str, default=DEFAULT_VQA_PROMPT,
                   help="Override the VQA prompt.")
    p.add_argument("--output-name", type=str, default="steer_vqa")
    return p.parse_args()


def generate_steered(
    model, processor, inputs, heads_by_layer, panel_positions,
    target_panel: int, n_query_heads: int, device: str,
    decode_only: bool, max_new_tokens: int,
    intervention: str, img_start: int, img_end: int, prompt_length: int,
    swap_bias: float,
) -> str:
    target_positions = panel_positions[target_panel]
    other_image_positions: list[int] = []
    for panel_idx in range(DEFAULT_N_PANELS):
        if panel_idx != target_panel:
            other_image_positions.extend(panel_positions[panel_idx])

    suppress_positions, boost_positions, pad_with_suppress = intervention_positions(
        mode=intervention,
        target_positions=target_positions,
        other_image_positions=other_image_positions,
        img_start=img_start,
        img_end=img_end,
        prompt_length=prompt_length,
    )

    hook_by_layer = {
        layer_idx: make_static_attention_mask_hook(
            head_indices=head_indices,
            suppress_positions=suppress_positions,
            boost_positions=boost_positions,
            n_query_heads=n_query_heads,
            device=device,
            swap_bias=swap_bias,
            decode_only=decode_only,
            pad_with_suppress=pad_with_suppress,
        )
        for layer_idx, head_indices in heads_by_layer.items()
    }

    mask_handles = register_mask_hooks(model, hook_by_layer)
    try:
        sequences = run_generation(model=model, inputs=inputs, max_new_tokens=max_new_tokens)
    finally:
        remove_handles(mask_handles)

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
    cutoff_score = float(np.percentile(gaze_scores, args.nongaze_percentile))
    random_heads = sample_non_gaze_heads(
        n_layers=n_layers,
        n_heads=n_query_heads,
        exclude=set(gaze_top),
        n_select=args.top_k_random,
        seed=args.seed,
        scores=gaze_scores,
        max_score=cutoff_score,
    )
    print(
        f"Non-gaze sampling: percentile={args.nongaze_percentile}, "
        f"cutoff_score={cutoff_score:.4f}, n_selected={len(random_heads)}",
        flush=True,
    )

    conditions = {
        f"gaze_top{args.top_k_gaze}": group_heads_by_layer(gaze_top),
        f"non_gaze_{args.top_k_random}": group_heads_by_layer(random_heads),
    }
    if args.include_all_heads:
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_query_heads)]
        conditions["all_heads"] = group_heads_by_layer(all_heads)

    comics_root = Path(args.comics_root)
    comic_dirs = list_comic_dirs(comics_root, n_panels=DEFAULT_N_PANELS)
    if args.comic_name:
        comic_dirs = [p for p in comic_dirs if p.name == args.comic_name]
    if args.start_comic_idx > 0:
        comic_dirs = comic_dirs[args.start_comic_idx :]
    if args.max_comics > 0:
        comic_dirs = comic_dirs[: args.max_comics]
    if not comic_dirs:
        raise FileNotFoundError(f"No valid {DEFAULT_N_PANELS}-panel comic directories under {comics_root}.")

    all_outcomes: dict[str, list[bool]] = {cond: [] for cond in conditions}
    junk_counts: dict[str, int] = {cond: 0 for cond in conditions}
    per_panel_outcomes: dict[str, dict[int, list[bool]]] = {
        cond: {p: [] for p in range(DEFAULT_N_PANELS)} for cond in conditions
    }

    # Flat per-generation log so results can be re-judged without re-running
    # the model.
    generations_jsonl_path = outputs.logs_dir / "generations.jsonl"
    generations_jsonl = open(generations_jsonl_path, "w")

    for sample_idx, comic_dir in enumerate(comic_dirs):
        strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
        print(f"\n=== Strip {sample_idx + 1}/{len(comic_dirs)}: {strip.name} ===", flush=True)

        inputs = prepare_inputs(processor, strip.strip, args.prompt, args.device)
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
        print(f"  baseline: {baseline_text}", flush=True)

        strip_results = {
            "strip_name": strip.name,
            "baseline_text": baseline_text,
            "conditions": {},
        }

        for cond_name, heads_by_layer in conditions.items():
            cond_results = []
            for target_panel in range(DEFAULT_N_PANELS):
                steered_text = generate_steered(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    heads_by_layer=heads_by_layer,
                    panel_positions=panel_positions,
                    target_panel=target_panel,
                    n_query_heads=n_query_heads,
                    device=args.device,
                    decode_only=args.decode_only,
                    max_new_tokens=args.max_new_tokens,
                    intervention=args.intervention,
                    img_start=img_start,
                    img_end=img_end,
                    prompt_length=int(inputs["input_ids"].shape[1]),
                    swap_bias=args.swap_bias,
                )

                try:
                    judgment = judge_match_target_panel(
                        strip_image=strip.strip,
                        segment_text=steered_text,
                        baseline_text=baseline_text,
                        target_panel=target_panel + 1,
                        n_panels=DEFAULT_N_PANELS,
                        model_name=args.claude_model,
                    )
                except Exception as exc:
                    print(f"  judge error: {exc}")
                    judgment = {
                        "matched_panel": None,
                        "is_junk": True,
                        "correct": False,
                        "reasoning": f"error: {exc}",
                    }

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

                generations_jsonl.write(json.dumps({
                    "strip_name": strip.name,
                    "comic_dir": str(comic_dir),
                    "condition": cond_name,
                    "target_panel": target_panel + 1,
                    "baseline_text": baseline_text,
                    "steered_text": steered_text,
                    "judgment": judgment,
                    "experiment": args.output_name,
                }) + "\n")
                generations_jsonl.flush()
                print(
                    f"  [{cond_name}] P{target_panel + 1}: {tag} | "
                    f"steered={steered_text.strip()[:90]}",
                    flush=True,
                )

                cond_results.append({
                    "target_panel": target_panel + 1,
                    "steered_text": steered_text,
                    "judgment": judgment,
                })
                write_text(
                    strip_log_dir / f"{cond_name}_target_{target_panel + 1}_text.txt",
                    steered_text,
                )

            strip_results["conditions"][cond_name] = cond_results

        dump_json(strip_log_dir / "summary.json", strip_results)

    generations_jsonl.close()

    print("\n" + "=" * 80, flush=True)
    print("AGGREGATE RESULTS  (does the steered text describe the target panel?)", flush=True)
    print("=" * 80, flush=True)

    aggregate = {
        "n_strips": len(comic_dirs),
        "intervention": args.intervention,
        "decode_only": args.decode_only,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "judge": "judge_match_target_panel",
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
    print(f"\nSaved aggregate results to {outputs.logs_dir / 'aggregate_results.json'}", flush=True)
    print(f"Saved per-generation JSONL to {generations_jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
