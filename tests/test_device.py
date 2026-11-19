import pytest
import torch

from ncm_iwsp.build import build_models
from ncm_iwsp.config import load_config
from ncm_iwsp.device import resolve_device
from ncm_iwsp.evaluate import evaluate_windows
from ncm_iwsp.packing import demo_texts, make_cache
from ncm_iwsp.pretok import CompactPretok
from ncm_iwsp.writers.stub import write_weights


def test_resolve_device_auto_and_cpu():
    auto = resolve_device("auto")
    if torch.cuda.is_available():
        assert auto.type == "cuda"
    else:
        assert auto.type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device(None).type == auto.type


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_evaluator_runs_on_cuda():
    cfg = load_config("configs/smoke.yaml")
    pretok = CompactPretok.from_config(cfg)
    cache = make_cache(demo_texts(), pretok, 16, hidden_pct=10, visible_pct=10, max_windows=4)
    _, vqvae, generator = build_models(cfg)
    write_weights(vqvae, generator, iteration=0)
    metrics = evaluate_windows(
        pretok,
        vqvae,
        generator,
        cache["visible"],
        train_windows=cache["visible_train"],
        batch_size=4,
        max_windows=4,
        n_generate=1,
        temperature=0.0,
        device="cuda",
    )
    assert metrics["device"].startswith("cuda")
    assert next(vqvae.parameters()).device.type == "cuda"
    assert next(generator.parameters()).device.type == "cuda"
    assert metrics["mean_level_ce"] > 0
    kinds = {row["kind"] for row in metrics["baselines"]["qualitative"]}
    assert "level0_unigram_plus_random_finer" in kinds
    assert "fixed_prompt" in kinds
