from ncm_iwsp.build import build_models, parameter_report
from ncm_iwsp.config import load_config

# Measured by instantiating the modules on CPU. Smoke matches the ncm-gpt1
# smoke template (count_params, embeddings included, codebooks included).
SMOKE_TOK = 58576
SMOKE_TOK_NON = 53968
SMOKE_GEN = 32992
SMOKE_GEN_NON = 31840
PILOT_TOK = 366240
PILOT_TOK_NON = 341664
PILOT_GEN = 187840
PILOT_GEN_NON = 183360


def test_smoke_matches_template_counts():
    cfg = load_config("configs/smoke.yaml")
    _, vqvae, generator = build_models(cfg)
    report = parameter_report(vqvae, generator)
    assert report["tokenizer"] == SMOKE_TOK
    assert report["tokenizer_non_embedding"] == SMOKE_TOK_NON
    assert report["generator"] == SMOKE_GEN
    assert report["generator_non_embedding"] == SMOKE_GEN_NON
    assert "GPT1" not in cfg["name"]


def test_pilot_counts():
    cfg = load_config("configs/pilot.yaml")
    _, vqvae, generator = build_models(cfg)
    report = parameter_report(vqvae, generator)
    assert report["tokenizer"] == PILOT_TOK
    assert report["tokenizer_non_embedding"] == PILOT_TOK_NON
    assert report["generator"] == PILOT_GEN
    assert report["generator_non_embedding"] == PILOT_GEN_NON
    total = report["tokenizer"] + report["generator"]
    assert 1e5 < total < 1e6
    assert len(vqvae.vq.levels) == 6
    assert vqvae.vq.levels[-1] == 64
    assert generator.n_layer == 2
    assert generator.n_embd == 64
