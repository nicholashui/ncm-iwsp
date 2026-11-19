"""Document-level splits, then pack short paragraphs into fixed windows.

MiniMind pretrain rows are short ``{"text": "..."}`` paragraphs. Padding each
one to ``L`` would waste the window. Documents are assigned to
``visible_train``, ``visible``, or ``hidden`` by a hash of the raw text
*before* packing, then each split is encoded, terminated with EOS, and packed
to length ``L``. Only the tail of the stream is padded.

The cache is a torch dict in the spirit of ``packed_windows.pt``:
``visible_train``, ``visible``, ``hidden``, and ``meta``. The MiniMind jsonl
itself is never written here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ncm_iwsp.pretok import CompactPretok


DEMO_DOCUMENTS = [
    "The small model writes its own weights.",
    "Pack several short paragraphs into one window.",
    "Random init is the baseline, not a trained model.",
    "A residual codebook keeps a coarse code and a fine code.",
    "Hidden documents must stay out of the writer.",
    "下一概念不是查表，而是一層一層的殘差碼。",
    "今天的試點只吃預訓練段落，不混進對話。",
    "This sentence is long enough to share a window with the next one when L is small.",
    "Evaluation ignores pad and does not call backward.",
    "Level zero is a unigram owned by the evaluator.",
    "Coarse to fine generation starts from that unigram.",
    "Do not rename this run NCM-GPT1.",
    "Byte tokens keep Chinese text lossless.",
    "ASCII smoke mode maps unknown characters to UNK.",
    "The audit rejects optimizers and weight downloads.",
    "A description has to say what the tensors mean.",
    "Short lines should not each become their own padded window.",
    "Visible validation is what the agent is allowed to see.",
    "Another plain English sentence about windows and codes.",
    "末段用來湊滿雜湊分桶，讓三個 split 都有文件。",
    "Windows are slices of a token stream, not one row per line.",
    "The generator predicts every level except the coarsest.",
    "Codebook usage is a diagnostic, not an EMA update.",
    "Fixed prompts are only a qualitative check.",
    "Success is beating random init and the unigram, honestly.",
    "No Kaggle quota from the BookCorpus run is spent here.",
    "Research code only. Read the dataset license before use.",
    "Sinusoidal embeddings are a closed form, not a fit.",
    "Each iteration rebuilds from the same seed, then overwrites.",
    "The scalar is the mean of the per-level cross entropies.",
]


def iter_jsonl_texts(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data file not found: {path}. The only MiniMind file this pilot "
            "accepts is pretrain_t2t_mini.jsonl, and it must not be committed."
        )
    name = path.name
    banned_names = {
        "pretrain_t2t.jsonl",
        "sft_t2t.jsonl",
        "sft_t2t_mini.jsonl",
        "dpo.jsonl",
        "rlaif.jsonl",
        "agent_rl.jsonl",
        "agent_rl_math.jsonl",
    }
    banned_bits = ("sft", "dpo", "rlaif", "agent_rl", "lora_")
    if name in banned_names or any(bit in name for bit in banned_bits):
        raise ValueError(
            f"refusing {name}: the pilot packs pretrain_t2t_mini.jsonl only, "
            "not the full pretrain file and not SFT, DPO, RLAIF, or tool data"
        )
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            text = obj.get("text", obj.get("content", ""))
            if isinstance(text, str) and text.strip():
                yield text


def document_split(text: str, hidden_pct: int, visible_pct: int) -> str:
    """Hash split on the raw document, independent of token packing."""
    hidden_pct = int(hidden_pct)
    visible_pct = int(visible_pct)
    if hidden_pct < 0 or visible_pct < 0 or hidden_pct + visible_pct >= 100:
        raise ValueError("hidden_pct and visible_pct must be non-negative and sum to < 100")
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < hidden_pct:
        return "hidden"
    if bucket < hidden_pct + visible_pct:
        return "visible"
    return "visible_train"


def split_documents(texts, hidden_pct: int, visible_pct: int) -> dict[str, list[str]]:
    groups = {"visible_train": [], "visible": [], "hidden": []}
    for text in texts:
        groups[document_split(text, hidden_pct, visible_pct)].append(text)
    return groups


def pack_documents(texts, pretok: CompactPretok, context_length: int, max_windows: int = 0) -> torch.Tensor:
    """Encode, append EOS after each document, pack to ``L``, pad only the tail."""
    eos = int(pretok.eos_token_id)
    pad = int(pretok.pad_token_id)
    length = int(context_length)
    if length < 2:
        raise ValueError("context_length must be at least 2")
    stream: list[int] = []
    cap = int(max_windows or 0)
    for text in texts:
        ids = list(pretok.encode(text) or [])
        if not ids:
            continue
        stream.extend(ids)
        stream.append(eos)
        if cap and len(stream) >= cap * length:
            break
    if not stream:
        raise ValueError("no tokens produced from the given documents")
    windows: list[list[int]] = []
    for start in range(0, len(stream), length):
        if cap and len(windows) >= cap:
            break
        chunk = stream[start : start + length]
        if len(chunk) < length:
            chunk = chunk + [pad] * (length - len(chunk))
        windows.append(chunk)
    if not windows:
        raise ValueError("packing produced no windows")
    return torch.tensor(windows, dtype=torch.long)


def nonpad_fraction(windows: torch.Tensor, pad_id: int) -> float:
    if windows.numel() == 0:
        return 0.0
    return float((windows != int(pad_id)).float().mean().item())


def make_cache(
    texts,
    pretok: CompactPretok,
    context_length: int,
    hidden_pct: int,
    visible_pct: int,
    max_windows: int = 0,
) -> dict:
    groups = split_documents(texts, hidden_pct, visible_pct)
    missing = [name for name, docs in groups.items() if not docs]
    if missing:
        raise ValueError(
            f"document split left {missing} empty. Lower the percentages or add documents. "
            "Splits are by document hash before packing, so a handful of rows can miss a bucket."
        )
    packed = {
        name: pack_documents(docs, pretok, context_length, max_windows=max_windows)
        for name, docs in groups.items()
    }
    stats = pretok.token_stats(texts)
    meta = {
        "context_length": int(context_length),
        "pad_id": int(pretok.pad_token_id),
        "eos_id": int(pretok.eos_token_id),
        "pretok_kind": pretok.kind,
        "vocab_size": int(pretok.vocab_size),
        "n_docs": {name: len(docs) for name, docs in groups.items()},
        "n_windows": {name: int(windows.shape[0]) for name, windows in packed.items()},
        "nonpad_fraction": {
            name: nonpad_fraction(windows, pretok.pad_token_id) for name, windows in packed.items()
        },
        "token_stats": stats,
        "hidden_pct": int(hidden_pct),
        "visible_pct": int(visible_pct),
        "format": "packed_windows",
    }
    return {**packed, "meta": meta}


def save_cache(cache: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("visible_train", "visible", "hidden", "meta"):
        if key not in cache:
            raise ValueError(f"cache {path} is missing '{key}'")
    return cache


def demo_texts() -> list[str]:
    return list(DEMO_DOCUMENTS)
