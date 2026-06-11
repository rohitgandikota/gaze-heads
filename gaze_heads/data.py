"""Comic-strip assembly from panel images.

Two source layouts are supported (see README.md):

1. Panel folders — one folder per comic with fixed panel files:

    <comics_root>/
      comic1/
        p1.png ... p6.png
      comic2/
        ...

2. The raw COMICS corpus (Iyyer et al. 2017) — one folder per comic with
   page-indexed panel files, from which strips are *sampled* as N consecutive
   panels of a random page:

    <comics_root>/
      <comic_id>/
        <page>_<panel>.jpg

In both cases panels are stitched into a single horizontal strip: resized to
a common height (preserving aspect ratio, dimensions rounded to a multiple of
32 so panel boundaries land cleanly on the vision-token grid) and
concatenated left to right with a thin white gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from gaze_heads.common import (
    DEFAULT_GAP,
    DEFAULT_N_PANELS,
    DEFAULT_TARGET_HEIGHT,
    IMAGE_SUFFIXES,
)


@dataclass
class ComicStrip:
    name: str
    strip: Image.Image
    panels: list[Image.Image]
    panel_widths: list[int]
    panel_paths: list[Path]
    target_height: int


def round_to_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, multiple * round(float(value) / float(multiple)))


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _find_panel_file(panel_dir: Path, panel_index: int) -> Optional[Path]:
    stem = f"p{panel_index}"
    for suffix in IMAGE_SUFFIXES:
        candidate = panel_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def list_comic_dirs(root: Path, n_panels: int = DEFAULT_N_PANELS) -> list[Path]:
    """List subdirectories of `root` that contain a complete p1..pN panel set."""
    if not root.exists():
        return []
    valid = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if all(_find_panel_file(path, idx) is not None for idx in range(1, n_panels + 1)):
            valid.append(path)
    return valid


def load_panels(panel_dir: Path, n_panels: int = DEFAULT_N_PANELS) -> list[tuple[Path, Image.Image]]:
    items: list[tuple[Path, Image.Image]] = []
    for panel_index in range(1, n_panels + 1):
        panel_path = _find_panel_file(panel_dir, panel_index)
        if panel_path is None:
            raise FileNotFoundError(
                f"Expected p1..p{n_panels} images in {panel_dir}, but panel {panel_index} is missing."
            )
        items.append((panel_path, _open_rgb(panel_path)))
    return items


def _resize_panels(
    images: list[Image.Image],
    target_height: int,
) -> tuple[list[Image.Image], list[int]]:
    resized: list[Image.Image] = []
    widths: list[int] = []
    target_height = round_to_multiple(target_height)
    for image in images:
        width, height = image.size
        new_width = round_to_multiple(int(width * target_height / max(height, 1)))
        resized_image = image.resize((new_width, target_height), Image.LANCZOS)
        resized.append(resized_image)
        widths.append(new_width)
    return resized, widths


def _assemble_strip(
    images: list[Image.Image],
    panel_widths: list[int],
    gap: int,
) -> Image.Image:
    total_width = int(sum(panel_widths))
    target_height = int(images[0].size[1])
    strip = Image.new("RGB", (total_width, target_height), (255, 255, 255))
    x_offset = 0
    for image in images:
        strip.paste(image, (x_offset, 0))
        x_offset += int(image.size[0])

    if gap > 0 and len(panel_widths) > 1:
        draw = ImageDraw.Draw(strip)
        x_offset = 0
        for width in panel_widths[:-1]:
            x_offset += width
            draw.rectangle(
                [x_offset - gap // 2, 0, x_offset + (gap - gap // 2) - 1, target_height - 1],
                fill="white",
            )
    return strip


def build_strip(
    panel_dir: Path,
    n_panels: int = DEFAULT_N_PANELS,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    gap: int = DEFAULT_GAP,
) -> ComicStrip:
    panel_items = load_panels(panel_dir, n_panels=n_panels)
    panel_paths = [path for path, _ in panel_items]
    raw_images = [image for _, image in panel_items]
    resized, widths = _resize_panels(raw_images, target_height=target_height)
    strip = _assemble_strip(resized, widths, gap=gap)
    return ComicStrip(
        name=panel_dir.name,
        strip=strip,
        panels=resized,
        panel_widths=widths,
        panel_paths=panel_paths,
        target_height=int(strip.size[1]),
    )


# ---------------------------------------------------------------------------
# Raw COMICS corpus: <root>/<comic_id>/<page>_<panel>.jpg
# ---------------------------------------------------------------------------


def list_raw_comic_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _group_raw_panels_by_page(comic_dir: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for path in sorted(comic_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.stem.split("_")
        if len(parts) != 2:
            continue
        try:
            page_num = int(parts[0])
            panel_num = int(parts[1])
        except ValueError:
            continue
        grouped.setdefault(page_num, []).append((panel_num, path))
    return {
        page_num: [path for _, path in sorted(items)]
        for page_num, items in sorted(grouped.items())
    }


def sample_raw_panel_paths(
    raw_panel_root: Path,
    rng: np.random.RandomState,
    n_panels: int = DEFAULT_N_PANELS,
) -> tuple[str, int, int, list[Path]]:
    """Sample (comic_id, page, start_idx, panel_paths): a random comic, a
    random interior page with >= n_panels panels, and a random consecutive
    panel window. Interior pages (skipping the first/last 5 of comics longer
    than 10 pages) avoid covers and back matter."""
    comic_ids = list_raw_comic_ids(raw_panel_root)
    if not comic_ids:
        raise FileNotFoundError(f"No comic directories found under {raw_panel_root}.")

    attempts = 0
    while attempts < 200:
        attempts += 1
        chosen_comic_id = comic_ids[int(rng.randint(len(comic_ids)))]
        comic_dir = raw_panel_root / chosen_comic_id
        grouped = _group_raw_panels_by_page(comic_dir)
        all_pages = sorted(grouped.keys())
        interior_pages = all_pages[5:-5] if len(all_pages) > 10 else all_pages
        eligible_pages = [page for page in interior_pages if len(grouped[page]) >= n_panels]
        if not eligible_pages:
            eligible_pages = [page for page, panels in grouped.items() if len(panels) >= n_panels]
        if not eligible_pages:
            continue
        page_num = int(eligible_pages[int(rng.randint(len(eligible_pages)))])
        panels = grouped[page_num]
        max_start = len(panels) - n_panels
        start_idx = int(rng.randint(max_start + 1)) if max_start > 0 else 0
        selection = panels[start_idx : start_idx + n_panels]
        if len(selection) == n_panels:
            return chosen_comic_id, page_num, start_idx, selection
    raise RuntimeError(f"Could not sample a valid {n_panels}-panel strip from {raw_panel_root}.")


def build_random_raw_strip(
    raw_panel_root: Path,
    rng: np.random.RandomState,
    n_panels: int = DEFAULT_N_PANELS,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    gap: int = DEFAULT_GAP,
) -> ComicStrip:
    comic_id, page_num, start_idx, panel_paths = sample_raw_panel_paths(
        raw_panel_root=raw_panel_root,
        rng=rng,
        n_panels=n_panels,
    )
    raw_images = [_open_rgb(path) for path in panel_paths]
    resized, widths = _resize_panels(raw_images, target_height=target_height)
    strip = _assemble_strip(resized, widths, gap=gap)
    return ComicStrip(
        name=f"{comic_id}_page{page_num}_start{start_idx}",
        strip=strip,
        panels=resized,
        panel_widths=widths,
        panel_paths=panel_paths,
        target_height=int(strip.size[1]),
    )
