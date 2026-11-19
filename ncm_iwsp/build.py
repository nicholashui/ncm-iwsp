"""Construct a seeded VQ-VAE and generator. Construction does not train."""

from __future__ import annotations

import torch

from ncm_iwsp.models.generator import NextConceptGenerator
from ncm_iwsp.models.vqvae import VQVAE
from ncm_iwsp.params import count_params
from ncm_iwsp.pretok import CompactPretok


def build_models(cfg: dict, seed: int | None = None):
    torch.manual_seed(int(cfg["seed"] if seed is None else seed))
    pretok = CompactPretok.from_config(cfg)
    vqvae = VQVAE(cfg["tokenizer"], pretok.vocab_size)
    generator = NextConceptGenerator(vqvae, cfg["generator"])
    return pretok, vqvae, generator


def lock_for_eval(module: torch.nn.Module) -> None:
    """Eval mode, and no parameter is left requesting gradients."""
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def parameter_report(vqvae, generator) -> dict:
    tok_total, tok_non = count_params(vqvae, exclude_params=("wte", "wpe"))
    gen_total, gen_non = count_params(generator, exclude_params=("wte", "wle", "wpe"))
    return {
        "tokenizer": tok_total,
        "tokenizer_non_embedding": tok_non,
        "generator": gen_total,
        "generator_non_embedding": gen_non,
        "tokenizer_millions": tok_total / 1e6,
        "generator_millions": gen_total / 1e6,
    }
