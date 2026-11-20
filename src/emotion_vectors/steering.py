"""Activation steering: add an emotion direction to one decoder layer.

The steering vector added at layer L is

    alpha * mean_token_norm[L] * v_unit

`v_unit` is the unit emotion vector at layer L. `mean_token_norm[L]` is the
typical L2 norm of a single residual-stream token at that layer, measured on
real neutral responses. Expressing strength as a fraction of that norm makes
`alpha` comparable across layers and across models.

`mean_token_norm` must be measured on *per-token* activations, not on the
mean-pooled story vectors: pooling averages away token-to-token variation and
shrinks the norm, which would make every alpha far too large.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

from .generation import _input_device, _set_seed, render_chat_prompt


def decoder_layers(model):
    """The list of decoder layer modules, e.g. `model.model.layers`."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "could not find decoder layers at model.model.layers; "
            "print(model) and update decoder_layers()"
        )
    return layers


@torch.no_grad()
def mean_token_norm(
    model,
    records: Sequence[dict],
    layer: int,
    *,
    sample_size: int = 20,
) -> float:
    """Typical per-token residual norm at `layer`, measured on neutral responses.

    `records` are rows from `generations.jsonl`; only `condition == "neutral"`
    rows are used, and only their response-token span.
    """
    device = _input_device(model)
    neutral = [
        record
        for record in records
        if record.get("condition") == "neutral"
        and record.get("validation", {}).get("valid", True)
    ][:sample_size]
    if not neutral:
        raise ValueError("no valid neutral records to measure the token norm on")

    norms: list[float] = []
    for record in neutral:
        ids = torch.tensor([record["full_ids"]], device=device)
        hidden = model(
            input_ids=ids, output_hidden_states=True, use_cache=False
        ).hidden_states
        # hidden[0] is the embedding output, so layer L is at index L + 1.
        span = hidden[layer + 1][0, record["response_start"] : record["response_end"]]
        if span.numel() == 0:
            continue
        norms.append(float(span.float().norm(dim=-1).mean()))
    if not norms:
        raise ValueError("every sampled neutral record had an empty response span")
    return sum(norms) / len(norms)


def _add_vector_hook(vector: torch.Tensor):
    """Forward hook that adds `vector` to a decoder layer's hidden-state output.

    Qwen decoder layers may return either a tensor or a tuple whose first
    element is the hidden state, so both shapes are handled.
    """

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vector.to(output[0].dtype),) + output[1:]
        return output + vector.to(output.dtype)

    return hook


@contextmanager
def steering(model, layer: int, vector: torch.Tensor | None) -> Iterator[None]:
    """Steer `layer` for the duration of the block. `vector=None` means no steering."""
    if vector is None:
        yield
        return
    handle = decoder_layers(model)[layer].register_forward_hook(
        _add_vector_hook(vector)
    )
    try:
        yield
    finally:
        handle.remove()


def steering_vector(
    unit_vector: torch.Tensor,
    alpha: float,
    token_norm: float,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """`alpha * token_norm * unit_vector`, moved to the model's device."""
    vector = unit_vector.float()
    vector = vector / vector.norm().clamp_min(1e-8)
    vector = alpha * token_norm * vector
    if device is not None:
        vector = vector.to(device)
    if dtype is not None:
        vector = vector.to(dtype)
    return vector


def random_control_vector(
    hidden_size: int,
    alpha: float,
    token_norm: float,
    *,
    seed: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """A random direction scaled to exactly the same norm as a real steering vector.

    Any text change this produces is an effect of perturbing the residual
    stream, not of the emotion direction specifically.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(hidden_size, generator=generator)
    return steering_vector(noise, alpha, token_norm, device=device, dtype=dtype)


@torch.no_grad()
def generate_steered(
    model,
    tokenizer,
    prompt: str,
    *,
    layer: int,
    vector: torch.Tensor | None = None,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    seed: int | None = None,
    thinking_enabled: bool = False,
) -> str:
    """Generate one response, optionally with a steering vector applied at `layer`.

    Greedy decoding (`do_sample=False`) is the default so that a difference
    between two outputs can only come from the steering vector.
    """
    text = render_chat_prompt(tokenizer, prompt, thinking_enabled=thinking_enabled)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
        _input_device(model)
    )
    if seed is not None:
        _set_seed(seed)
    with steering(model, layer, vector):
        output = model.generate(
            **encoded,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def check_zero_alpha(
    model,
    tokenizer,
    prompt: str,
    *,
    layer: int,
    hidden_size: int,
    **kwargs,
) -> bool:
    """A zero vector must reproduce the unsteered output exactly.

    If this fails, the hook is wired to the wrong module or is mutating state,
    and every steering result below it is meaningless.
    """
    device = _input_device(model)
    baseline = generate_steered(
        model, tokenizer, prompt, layer=layer, vector=None, **kwargs
    )
    zeroed = generate_steered(
        model,
        tokenizer,
        prompt,
        layer=layer,
        vector=torch.zeros(hidden_size, device=device),
        **kwargs,
    )
    return baseline == zeroed
