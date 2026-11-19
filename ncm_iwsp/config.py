"""YAML configs for the two pilot rungs. No optimizer section on purpose."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    _validate(cfg, path)
    return cfg


def _validate(cfg: dict, path: Path) -> None:
    for key in ("name", "seed", "pretok", "tokenizer", "generator", "data"):
        if key not in cfg:
            raise ValueError(f"{path} is missing '{key}'")
    levels = [int(x) for x in cfg["tokenizer"]["levels"]]
    if levels[-1] != int(cfg["tokenizer"]["context_length"]):
        raise ValueError("tokenizer.levels[-1] must equal context_length")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError(f"VQ levels must be strictly increasing, got {levels}")
    if cfg["generator"].get("predict_level0", False):
        raise ValueError("this pilot pins predict_level0: false, as in official NCM")
    if cfg["generator"].get("attn_pattern", "block_diagonal") != "block_diagonal":
        raise ValueError("this pilot pins block_diagonal attention")
    name = str(cfg["name"])
    if "GPT1" in name or "gpt1" in name:
        raise ValueError("do not name an iWSP config NCM-GPT1")
