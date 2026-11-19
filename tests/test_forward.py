import torch

from ncm_iwsp.build import build_models, lock_for_eval
from ncm_iwsp.config import load_config
from ncm_iwsp.packing import pack_documents
from ncm_iwsp.pretok import CompactPretok


def _windows(cfg, texts):
    pretok = CompactPretok.from_config(cfg)
    return pretok, pack_documents(texts, pretok, cfg["tokenizer"]["context_length"])


def test_smoke_forward_on_synthetic_windows():
    cfg = load_config("configs/smoke.yaml")
    pretok, vqvae, generator = build_models(cfg)
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    _, windows = _windows(cfg, ["hello world", "pack these words", "a third short line"])
    with torch.no_grad():
        logits, indices = vqvae.reconstruct_logits(windows)
        ncm = vqvae.vq.idx_to_ncm_input(indices)
        prefix = torch.full((windows.shape[0], 1), pretok.eos_token_id, dtype=torch.long)
        teacher = torch.cat([vqvae.encoder.wte(prefix), ncm], dim=1)
        gen_logits = generator(teacher)
    assert indices.shape[1] == sum(vqvae.vq.levels) == 1 + 4 + 8 + 16
    assert logits.shape == (windows.shape[0], 16, pretok.vocab_size)
    assert gen_logits.shape == (windows.shape[0], 4 + 8 + 16, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(gen_logits).all()
    level0 = indices[:, :1]
    generated = generator.generate(level0, prefix, temperature=0.0)
    assert generated.shape == (windows.shape[0], 29)


def test_pilot_forward_on_synthetic_windows():
    cfg = load_config("configs/pilot.yaml")
    pretok, vqvae, generator = build_models(cfg)
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    text = "下一概念 " * 8
    _, windows = _windows(cfg, [text, text + "more"])
    with torch.no_grad():
        _, indices = vqvae.reconstruct_logits(windows[:2])
        ncm = vqvae.vq.idx_to_ncm_input(indices)
        prefix = torch.zeros(indices.shape[0], 1, dtype=torch.long)
        gen_logits = generator(torch.cat([vqvae.encoder.wte(prefix), ncm], dim=1))
    assert indices.shape[1] == sum([1, 4, 8, 16, 32, 64])
    assert gen_logits.shape[1] == sum([4, 8, 16, 32, 64])
    assert gen_logits.shape[-1] == 256
    assert torch.isfinite(gen_logits).all()
    assert pretok.vocab_size == 320
