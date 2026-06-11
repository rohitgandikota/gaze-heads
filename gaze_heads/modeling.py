"""Qwen3-VL model loading, input preparation, and generation helpers.

Everything here assumes the Qwen3-VL family (2B / 4B / 8B / 32B Instruct).
The model is always loaded with eager attention so per-head attention weights
are accessible, and so the attention-mask steering hooks see the mask kwarg.
"""
from __future__ import annotations

from typing import Any

import torch

from gaze_heads.common import DEFAULT_MODEL_ID

# Qwen-VL expands the image into this many-per-patch placeholder token.
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_PAD_TOKEN_ID_FALLBACK = 151655


def load_model_and_processor(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Load a Qwen3-VL model + processor with eager attention."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    use_bfloat16 = device.startswith("cuda") and torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, processor


def prepare_inputs(processor: Any, image: Any, prompt: str, device: str) -> Any:
    """Build model inputs for a single (image, text) user message."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def get_image_token_id(processor: Any) -> int:
    tokenizer = processor.tokenizer
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if token_id is not None and token_id >= 0:
        return int(token_id)
    return IMAGE_PAD_TOKEN_ID_FALLBACK


def find_image_token_range(inputs: Any, processor: Any) -> tuple[int, int]:
    """Return [start, end) positions of the image-pad tokens in the prompt."""
    image_token_id = get_image_token_id(processor)
    token_ids = inputs["input_ids"][0].tolist()
    first = None
    last = None
    for idx, token_id in enumerate(token_ids):
        if token_id == image_token_id:
            if first is None:
                first = idx
            last = idx
    if first is None or last is None:
        raise ValueError(
            f"No image tokens (id={image_token_id}) found in prompt inputs. "
            "The chat template did not expand the image into placeholder tokens."
        )
    return int(first), int(last + 1)


def extract_prefill_attentions(model: Any, inputs: Any) -> Any:
    with torch.no_grad():
        return model(**inputs, output_attentions=True, return_dict=True)


def run_generation(
    model: Any,
    inputs: Any,
    max_new_tokens: int,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> torch.Tensor:
    """Greedy generation. The optional repetition controls exist for dynamic
    narration steering, where hard +/-inf attention biases under pure greedy
    decoding can lock the KV cache into repetition loops."""
    extra: dict[str, Any] = {}
    if repetition_penalty is not None:
        extra["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size is not None:
        extra["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **extra,
        )
    return out


def decode_generated_text(processor: Any, sequences: torch.Tensor, input_length: int) -> str:
    generated = sequences[0, input_length:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def decode_generated_tokens(processor: Any, sequences: torch.Tensor, input_length: int) -> list[str]:
    generated = sequences[0, input_length:]
    return [processor.tokenizer.decode([token_id]) for token_id in generated.tolist()]


def model_dims(model: Any) -> tuple[int, int, int]:
    """Return (n_lm_layers, n_attention_heads_per_layer, vision_spatial_merge)."""
    cfg = model.config
    n_layers = int(cfg.text_config.num_hidden_layers)
    n_heads = int(cfg.text_config.num_attention_heads)
    spatial_merge = int(getattr(cfg.vision_config, "spatial_merge_size", 2))
    return n_layers, n_heads, spatial_merge


def language_model_layers(model: Any) -> torch.nn.ModuleList:
    """Return the LM transformer blocks where gaze heads live.

    Transformers releases have nested the Qwen-VL language model differently,
    so we try the known paths and accept the first that yields a ModuleList.
    """
    candidate_paths = [
        ("model", "language_model", "layers"),
        ("model", "model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
    ]
    for path in candidate_paths:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise AttributeError(f"Could not locate Qwen3-VL LM layers; tried {candidate_paths}")
