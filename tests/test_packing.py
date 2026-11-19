import json

import pytest
import torch

from ncm_iwsp.packing import (
    document_split,
    iter_jsonl_texts,
    make_cache,
    pack_documents,
    split_documents,
)
from ncm_iwsp.pretok import CompactPretok


def test_short_paragraphs_share_a_window():
    pretok = CompactPretok("ascii", 128)
    windows = pack_documents(["abcd", "efgh"], pretok, context_length=16)
    assert windows.shape == (1, 16)
    # 4 chars + EOS + 4 chars + EOS, then pad. Not one padded window per paragraph.
    assert int((windows[0] != pretok.pad_token_id).sum()) == 10
    assert int(windows[0, -1]) == pretok.pad_token_id
    eos_positions = (windows[0] == pretok.eos_token_id).nonzero().flatten().tolist()
    assert eos_positions == [4, 9]


def test_long_stream_is_sliced_not_truncated_to_one_doc():
    pretok = CompactPretok("ascii", 128)
    windows = pack_documents(["a" * 20, "b" * 20], pretok, context_length=16)
    assert windows.shape[0] >= 2
    assert windows.shape[1] == 16
    # The second window continues the stream; it is not a fresh pad of the second paragraph alone.
    assert int(windows[1, 0]) != pretok.pad_token_id


def test_document_split_is_stable_and_exclusive():
    assert document_split("same text", 10, 10) == document_split("same text", 10, 10)
    groups = split_documents(["alpha text", "beta text", "gamma text"], 10, 10)
    seen = []
    for docs in groups.values():
        seen.extend(docs)
    assert sorted(seen) == ["alpha text", "beta text", "gamma text"]
    assert sum(len(docs) for docs in groups.values()) == 3


def test_hidden_marker_cannot_appear_in_a_visible_window():
    pretok = CompactPretok("ascii", 128)
    hidden = pack_documents(["UNIQUEHIDDENMARKER"], pretok, 32)
    visible = pack_documents(["hello there friend", "another visible paragraph"], pretok, 32)
    hidden_ids = set(hidden.reshape(-1).tolist())
    # The marker's characters are not required to be globally unique ids, so compare the packed stream.
    marker = pretok.encode("UNIQUEHIDDENMARKER") + [pretok.eos_token_id]
    visible_stream = visible.reshape(-1).tolist()
    for start in range(len(visible_stream) - len(marker) + 1):
        assert visible_stream[start : start + len(marker)] != marker
    assert pretok.eos_token_id in hidden_ids


def test_jsonl_and_cache_roundtrip(tmp_path):
    path = tmp_path / "notes.jsonl"
    rows = [{"text": "one short paragraph"}, {"text": "two short paragraph"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    texts = list(iter_jsonl_texts(path))
    assert texts == ["one short paragraph", "two short paragraph"]
    pretok = CompactPretok("ascii", 128)
    many = [f"paragraph number {i} is its own document" for i in range(40)]
    cache = make_cache(many, pretok, 16, hidden_pct=10, visible_pct=10, max_windows=8)
    assert set(cache) >= {"visible_train", "visible", "hidden", "meta"}
    for key in ("visible_train", "visible", "hidden"):
        assert cache[key].dtype == torch.long
        assert cache[key].shape[1] == 16
        assert cache[key].shape[0] >= 1
    assert cache["meta"]["format"] == "packed_windows"
    assert cache["meta"]["nonpad_fraction"]["visible"] > 0


def test_refuses_sft_and_full_pretrain_names(tmp_path):
    for name in ("sft_t2t_mini.jsonl", "dpo.jsonl", "pretrain_t2t.jsonl", "rlaif.jsonl"):
        path = tmp_path / name
        path.write_text('{"text": "no"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            list(iter_jsonl_texts(path))
