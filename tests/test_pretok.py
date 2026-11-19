from ncm_iwsp.pretok import CompactPretok


def test_ascii_roundtrip_and_small_vocab():
    pretok = CompactPretok("ascii", pad_vocab_to=128)
    text = "Hello, world.\n"
    assert pretok.decode(pretok.encode(text)) == text
    assert pretok.vocab_size == 128
    assert pretok.used_vocab_size < 128
    assert pretok.vocab_size not in (6400, 50257, 50304)
    # CJK is not in the ASCII table, so it becomes UNK and does not round-trip.
    assert pretok.encode("你") == [pretok.unk_token_id]


def test_byte_pretok_is_lossless_for_chinese():
    pretok = CompactPretok("byte", pad_vocab_to=320)
    text = "下一概念"
    ids = pretok.encode(text)
    assert pretok.decode(ids) == text
    assert len(ids) == len(text.encode("utf-8")) == 12
    stats = pretok.token_stats([text, "Hello"])
    assert stats["vocab_size"] == 320
    assert stats["used_vocab_size"] == 259
    assert stats["vocab_size"] not in (6400, 50304)
    # 4 CJK characters -> 12 bytes, plus 5 ASCII letters.
    assert stats["cjk_chars"] == 4
    assert abs(stats["tokens_per_char"] - (12 + 5) / (4 + 5)) < 1e-9
