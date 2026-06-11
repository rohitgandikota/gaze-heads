"""Claude-as-judge scoring for steering experiments.

The judge sees the comic strip plus the steered text and is forced to pick
the single panel (1..N) the text best matches; junk/degenerate outputs are
flagged and count as misses. Requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

import numpy as np

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


def require_api_key() -> None:
    """Fail fast before a long GPU run if judging is impossible."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is required for LLM-judged evaluation. "
            "Export it before running, e.g. export ANTHROPIC_API_KEY=sk-ant-..."
        )


def _get_client():
    require_api_key()
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _create_with_retry(client, *, retries: int = 6, base_delay: float = 2.0, **kwargs):
    """Call client.messages.create with exponential backoff for transient errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = (
                "overload" in msg
                or "rate" in msg
                or "429" in msg
                or "529" in msg
                or "timeout" in msg
                or "503" in msg
            )
            if not transient:
                raise
            time.sleep(base_delay * (2 ** attempt))
    if last_exc is not None:
        raise last_exc


def _extract_json(response_text: str) -> dict:
    for pattern in [r"\{[\s\S]*?\}", r"\{[\s\S]*\}"]:
        for match in re.finditer(pattern, response_text):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError("LLM judge response did not contain a valid JSON object.")


def _encode_image_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _normalize_for_match(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^panel\s*\d+\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def steered_matches_baseline(steered: str, baseline: str, jaccard_threshold: float = 0.85) -> bool:
    """Detect when steering had essentially zero effect (the non-gaze control
    signature): normalized exact match or word-level Jaccard >= threshold."""
    if baseline is None:
        return False
    a = _normalize_for_match(steered)
    b = _normalize_for_match(baseline)
    if not a or not b:
        return False
    if a == b:
        return True
    a_set = set(a.split())
    b_set = set(b.split())
    if not a_set or not b_set:
        return False
    union = len(a_set | b_set)
    if union == 0:
        return False
    return (len(a_set & b_set) / union) >= jaccard_threshold


def judge_match_target_panel(
    strip_image,
    segment_text: str,
    baseline_text: str | None,
    target_panel: int,
    n_panels: int = 6,
    model_name: str = DEFAULT_JUDGE_MODEL,
    treat_baseline_match_as_miss: bool = True,
) -> dict[str, Any]:
    """Forced-choice judge: which of the N panels does the text best match?

    CORRECT iff matched_panel == target_panel. If the steered text is nearly
    identical to the baseline (Jaccard >= 0.85), the sample is marked MISS
    without calling the model — steering that changes nothing never counts
    as a hit.
    """
    if (
        treat_baseline_match_as_miss
        and baseline_text is not None
        and steered_matches_baseline(segment_text, baseline_text)
    ):
        return {
            "matched_panel": None,
            "is_junk": False,
            "correct": False,
            "matches_baseline": True,
            "reasoning": "steered text is essentially identical to the baseline (no steering effect)",
        }

    client = _get_client()
    image_b64 = _encode_image_base64(strip_image)
    baseline_blurb = (
        f'Baseline answer (no intervention): "{baseline_text}"\n\n'
        if baseline_text is not None
        else ""
    )
    message = _create_with_retry(
        client,
        model=model_name,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is a {n_panels}-panel comic strip "
                            f"(panels numbered 1 to {n_panels} from left "
                            "to right).\n\n"
                            f"{baseline_blurb}"
                            f'Steered answer: "{segment_text}"\n\n'
                            "IMPORTANT: Ignore any panel numbering inside "
                            "the steered answer ('Panel 1:', 'Panel 2:'). "
                            "The model often numbers sequentially "
                            "regardless of which panel it is actually "
                            "describing. Match by VISUAL CONTENT.\n\n"
                            "Pick exactly one panel (an integer "
                            f"1..{n_panels}) whose visual content the "
                            "steered answer best describes — the action, "
                            "objects, or scene that uniquely identifies "
                            "that panel.\n\n"
                            "If the answer is incoherent, repetitive, "
                            "degenerate, empty, or just numbers/labels, "
                            "set is_junk=true and matched_panel=null.\n\n"
                            "If the answer is generic and could fit "
                            "multiple panels, still pick whichever panel "
                            "the answer best matches based on any "
                            "specific content present.\n\n"
                            "Return ONLY a JSON object:\n"
                            f'{{"matched_panel": <1..{n_panels} or null>, '
                            '"is_junk": <true/false>, '
                            '"reasoning": "<one sentence>"}}'
                        ),
                    },
                ],
            }
        ],
    )
    result = _extract_json(message.content[0].text.strip())
    is_junk = bool(result.get("is_junk", False))
    raw_matched = result.get("matched_panel", None)
    if isinstance(raw_matched, bool):
        matched_panel = None
    else:
        try:
            matched_panel = int(raw_matched) if raw_matched is not None else None
        except (TypeError, ValueError):
            matched_panel = None
    if matched_panel is not None and not (1 <= matched_panel <= n_panels):
        matched_panel = None

    correct = (
        not is_junk
        and matched_panel is not None
        and matched_panel == int(target_panel)
    )
    return {
        "matched_panel": matched_panel,
        "is_junk": is_junk,
        "correct": correct,
        "matches_baseline": False,
        "reasoning": str(result.get("reasoning", "")),
    }


def bootstrap_ci(
    outcomes: list[bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Accuracy with a bootstrap confidence interval."""
    if not outcomes:
        return {"accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    outcomes_arr = np.array(outcomes, dtype=np.float64)
    accuracy = float(outcomes_arr.mean())
    n = len(outcomes_arr)

    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = outcomes_arr[indices].mean(axis=1)

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return {"accuracy": accuracy, "ci_low": ci_low, "ci_high": ci_high, "n": n}
