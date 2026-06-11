#!/usr/bin/env python3
"""Download the OpenAI comic-strips dataset from Hugging Face and export it
into the panel-folder layout the scripts expect (comicN/p1.png..p6.png).

Usage:
    python download_data.py                      # writes to data/comics
    python download_data.py --out /path/to/dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gaze_heads.common import DEFAULT_COMICS_ROOT

DATASET_ID = "baulab/openai-comic-strips"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the OpenAI comic strips from Hugging Face.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_COMICS_ROOT),
                        help="Output folder (default: data/comics, the default --comics-root).")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    args = parser.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset_id, split="train")

    for row in ds:
        comic_dir = out_root / f"comic{row['comic_id']}"
        comic_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 7):
            row[f"panel_{k}"].save(comic_dir / f"p{k}.png")

    print(f"Exported {len(ds)} comics to {out_root}")


if __name__ == "__main__":
    main()
