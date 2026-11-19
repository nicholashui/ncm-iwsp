import csv
import json

import torch

from ncm_iwsp.build import build_models
from ncm_iwsp.config import load_config
from ncm_iwsp.loop import run_dry_loop
from ncm_iwsp.packing import demo_texts, make_cache, save_cache
from ncm_iwsp.pretok import CompactPretok
from ncm_iwsp.writers.stub import write_weights


def test_stub_noop_matches_seed_and_later_iteration_writes():
    cfg = load_config("configs/smoke.yaml")
    _, left, left_gen = build_models(cfg)
    _, right, right_gen = build_models(cfg)
    write_weights(left, left_gen, iteration=0)
    assert torch.equal(left.encoder.wte.weight, right.encoder.wte.weight)
    write_weights(right, right_gen, iteration=2)
    assert not torch.equal(left.encoder.wte.weight, right.encoder.wte.weight)
    # Axis-aligned codebook: one 1-valued entry per row.
    codebook = right.vq.embeddings[0].weight
    assert torch.count_nonzero(codebook) == codebook.shape[0]


def test_dry_loop_writes_visible_csv_and_hidden_log(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    pretok = CompactPretok.from_config(cfg)
    cache = make_cache(demo_texts(), pretok, 16, hidden_pct=10, visible_pct=10, max_windows=4)
    packed = tmp_path / "packed.pt"
    save_cache(cache, packed)
    out = tmp_path / "run"
    summary = run_dry_loop(cfg, cache, "ncm_iwsp/writers/stub.py", out, iterations=2)
    assert summary["status"] == "ok"
    assert summary["param_counts"]["tokenizer"] == 58576
    with (out / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in rows] == ["0", "1"]
    assert all(row["audit"] == "pass" for row in rows)
    assert "hidden" not in (out / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    hidden_lines = (out / "hidden_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(hidden_lines) == 2
    assert json.loads(hidden_lines[0])["operator_only"] is True
    baselines = json.loads((out / "baselines.json").read_text(encoding="utf-8"))
    assert "random_init" in baselines
    assert baselines["most_frequent_token_ce"] > 0
    assert baselines["token_unigram_nll"] > 0
    assert baselines["level0_unigram_nll"] > 0
    assert (out / "snapshots" / "iter_000.py").exists()
    assert (out / "snapshots" / "iter_001.py").exists()
