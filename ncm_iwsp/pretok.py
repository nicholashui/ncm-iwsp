"""Frozen compact pretokensizers. These are encoding rules, not trained vocabs.

Two kinds:

* ``ascii`` — printable ASCII plus newline/tab, padded to a small table
  (Rung 0 uses 128, the same size as the ncm-gpt1 smoke char vocab).
  Characters outside that set, including CJK, become UNK.
* ``byte`` — UTF-8 bytes. Vocab is 3 specials + 256 bytes, optionally padded.
  Chinese is lossless at 3 tokens per character. This is not MiniMind's
  trained 6400-token vocabulary and not GPT-2's 50257/50304 BPE.
"""

from __future__ import annotations

# Printable ASCII plus newline and tab. Specials occupy ids 0-2.
_ASCII = [chr(i) for i in range(32, 127)] + ["\n", "\t"]


class CompactPretok:
    eos_token = "<|endoftext|>"
    pad_token = "<|pad|>"
    unk_token = "<|unk|>"

    def __init__(self, kind: str = "ascii", pad_vocab_to: int = 128):
        kind = str(kind)
        if kind not in ("ascii", "byte"):
            raise ValueError(f"pretok kind must be 'ascii' or 'byte', got {kind!r}")
        self.kind = kind
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2
        if kind == "ascii":
            self.itos = [self.eos_token, self.pad_token, self.unk_token] + list(_ASCII)
            self.stoi = {ch: i for i, ch in enumerate(self.itos)}
            self.used_vocab_size = len(self.itos)
        else:
            # ids 3..258 are the 256 UTF-8 byte values. No trained merges.
            self.itos = None
            self.stoi = None
            self.used_vocab_size = 3 + 256
        pad = int(pad_vocab_to)
        if pad < self.used_vocab_size:
            raise ValueError(
                f"pad_vocab_to={pad} is smaller than the used vocab {self.used_vocab_size}"
            )
        self.vocab_size = pad

    @classmethod
    def from_config(cls, cfg: dict) -> "CompactPretok":
        block = cfg["pretok"] if "pretok" in cfg else cfg
        return cls(kind=block.get("kind", "ascii"), pad_vocab_to=int(block["pad_vocab_to"]))

    def encode(self, text: str) -> list[int]:
        if self.kind == "ascii":
            unk = self.unk_token_id
            return [self.stoi.get(ch, unk) for ch in text]
        raw = text.encode("utf-8")
        return [3 + byte for byte in raw]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        specials = {self.eos_token_id, self.pad_token_id, self.unk_token_id}
        if self.kind == "ascii":
            out = []
            for i in ids:
                i = int(i)
                if i in specials:
                    continue
                if 0 <= i < self.used_vocab_size:
                    out.append(self.itos[i])
            return "".join(out)
        body = bytearray()
        for i in ids:
            i = int(i)
            if i in specials:
                continue
            byte = i - 3
            if 0 <= byte <= 255:
                body.append(byte)
        return body.decode("utf-8", errors="replace")

    def token_stats(self, texts) -> dict:
        """Closed-form length stats. Not a measurement of the MiniMind corpus."""
        chars = 0
        toks = 0
        cjk_chars = 0
        ascii_chars = 0
        for text in texts:
            chars += len(text)
            toks += len(self.encode(text))
            for ch in text:
                code = ord(ch)
                if 0x4E00 <= code <= 0x9FFF:
                    cjk_chars += 1
                elif ch.isascii() and not ch.isspace():
                    ascii_chars += 1
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "used_vocab_size": self.used_vocab_size,
            "chars": chars,
            "tokens": toks,
            "chars_per_token": (chars / toks) if toks else None,
            "tokens_per_char": (toks / chars) if chars else None,
            "cjk_chars": cjk_chars,
            "ascii_letters": ascii_chars,
            "note": (
                "byte mode uses one token per UTF-8 byte (3 tokens per CJK character). "
                "ascii mode maps CJK to UNK."
                if self.kind == "byte"
                else "ascii mode is one token per listed character and maps CJK to UNK."
            ),
        }
