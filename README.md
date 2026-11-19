# NCM-MiniMind-iWSP

Research code for a **from-scratch** Next-Concept Model whose weights are written in Python, not fit by backpropagation. The pilot name is **NCM-MiniMind-iWSP-pilot**. It is not NCM-GPT1.

這是研究程式：用程式直接寫下一概念模型（NCM）的權重，不使用反向傳播。試點名稱是 **NCM-MiniMind-iWSP-pilot**。它不是 NCM-GPT1，也不接 BookCorpus 的梯度訓練。

The method follows interpretable weight-space programming ([arXiv:2611.18114](https://arxiv.org/abs/2611.18114)): a writer fills `write_weights`, a **fixed** evaluator returns one scalar, and an audit rejects training. The network is the NCM hierarchy from [HazyResearch/ncm](https://github.com/HazyResearch/ncm) ([arXiv:2610.14162](https://arxiv.org/html/2610.14162v1)): a residual multiscale VQ-VAE and a coarse-to-fine generator. Level 0 is not predicted by the transformer. The evaluator owns that unigram.

Official gradient BookCorpus training stays in [nicholashui/ncm-gpt1](https://github.com/nicholashui/ncm-gpt1) under the name **NCM-GPT1**. This repository does not modify that repo, does not resume its checkpoints, and must not spend its Kaggle T4 quota unless Nicholas reallocates it.

## License

This harness is MIT. Portions of `ncm_iwsp/models/` are adapted from HazyResearch/ncm commit `c70274273f489abc73832b2b98584a8483b6b203` (MIT, Copyright (c) 2025 hermann kumbong). Codebook EMA, k-means restarts, and the official `train_tokenizer.py` / `train_generator.py` loops were not copied.

The MiniMind **code** is Apache-2.0. The dataset card for [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) lists **both** `apache-2.0` and `cc-by-nc-2.0`. CC-BY-NC-2.0 is non-commercial. The card does not say which file is under which license. This repository does not grant legal clearance. It is research code. Do not commit `pretrain_t2t_mini.jsonl` or any other corpus file.

資料集卡片同時標了 Apache-2.0 和 CC-BY-NC-2.0。CC-BY-NC-2.0 是非商業授權。本倉庫沒有做法律判斷，只供研究。不要把語料檔提交進 git。

## No backpropagation

The writer may assign tensors in closed form from a deterministic seed. The default iteration 0 is a no-op and leaves the seeded initialization in place. That random initialization is the baseline.

Rejected in the writer file, matched on code rather than comments:

- `loss.backward`, `torch.optim`, Adam / AdamW, `sklearn` `.fit`
- official `train_tokenizer` / `train_generator`
- codebook EMA (the quantizer forward never updates embeddings)
- `from_pretrained`, `torch.hub`, `torch.load`, and other weight downloads
- a `write_weights` signature that accepts data, labels, or the hidden split
- a missing description of what the tensors are

`pytest` also scans this package for the same training calls. A scan is not a proof; it is the same kind of post-hoc check iWSP uses because the writer’s own Python is free to cheat.

## What is pinned

The architecture stays a residual VQ-VAE plus a coarse-to-fine generator. Replacing that with a fact-lookup table is out of scope, even if a table scored better. The writer may change sizes only by editing a config that still builds these two modules. The pretok is a frozen encode/decode rule. It is not passed into `write_weights`.

`write_weights(tokenizer, generator, iteration=0)` receives the **VQ-VAE module** as `tokenizer` (`encoder.wte`, `vq.embeddings`, the level-0 sampler, `quant_resi`, `decoder`) and the generator. It does not receive windows.

Pretok choices, both a few hundred ids, neither MiniMind’s trained 6400-token vocab nor GPT-2’s 50,304-row table:

| Kind | Where | What a character costs |
| --- | --- | --- |
| `ascii` | Rung 0 smoke, vocab padded to 128 | one id per listed ASCII character; CJK becomes UNK and does not round-trip |
| `byte` | Rung 1 pilot, vocab padded to 320 (259 used: 3 specials + 256 UTF-8 bytes) | one id per byte. `下一概念` is 4 characters and 12 tokens |

## Measured parameter counts

Counted on CPU with `count_params` after instantiation. Codebooks are included even though `requires_grad` is false. Non-embedding totals drop names containing `wte`, `wpe`, or (generator) `wle`, matching the official helper. There is no training in these numbers.

| Config | Context | VQ levels | Codebook | Embed | Generator | Tokenizer params | Generator params |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `configs/smoke.yaml` | 16 | 4: `[1, 4, 8, 16]` | 64 | 32 | 2 × 32 × 4 | **58,576** (0.058576M); non-embedding 53,968 | **32,992** (0.032992M); non-embedding 31,840 |
| `configs/pilot.yaml` | 64 | 6: `[1, 4, 8, 16, 32, 64]` | 256 | 64 | 2 × 64 × 4 | **366,240** (0.366240M); non-embedding 341,664 | **187,840** (0.187840M); non-embedding 183,360 |

Smoke matches the sizes measured for the ncm-gpt1 `config/smoke.yaml` template (58,576 + 32,992). Pilot total is 554,080 parameters. Reproduce with:

```bash
python -m ncm_iwsp count --config configs/smoke.yaml
python -m ncm_iwsp count --config configs/pilot.yaml
```

Both rungs use `enc_layers=0` (embedding lookup), `dec_layers=1`, four decoder heads, group-norm before quantization, the learned stride cascade on level 0, Phi residual blend `quant_resi` at ratio 0.5, block-diagonal attention, RoPE, level encodings, and `predict_level0: false`. A local weight-copy check against the ncm-gpt1 smoke modules matched encode indices and greedy generation exactly. That checkout is not a dependency of this repo.

## Install

CPU is enough for the smoke dry loop.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest
```

## Run the dry loop

Synthetic paragraphs are built in. They are not MiniMind text. `--demo` forces at least a 10% / 10% document split so the short list fills every bucket. A real pack uses the percentages in the config (pilot yaml is 1% hidden, 1% visible).

```bash
python -m ncm_iwsp pack --demo --config configs/smoke.yaml --out cache/smoke.pt
python -m ncm_iwsp eval --config configs/smoke.yaml --packed cache/smoke.pt --iteration 0
python -m ncm_iwsp loop --config configs/smoke.yaml --packed cache/smoke.pt --iters 3 --out runs/smoke
```

Rung 1, still without the corpus, using the byte pretok:

```bash
python -m ncm_iwsp pack --demo --config configs/pilot.yaml --out cache/pilot-demo.pt
python -m ncm_iwsp loop --config configs/pilot.yaml --packed cache/pilot-demo.pt --iters 3 --out runs/pilot-demo
```

Real pretrain, one file only, not committed:

```bash
hf download jingyaogong/minimind_dataset pretrain_t2t_mini.jsonl \
  --repo-type dataset --local-dir dataset
python -m ncm_iwsp pack \
  --config configs/pilot.yaml \
  --jsonl dataset/pretrain_t2t_mini.jsonl \
  --out cache/pilot.pt
python -m ncm_iwsp loop \
  --config configs/pilot.yaml \
  --packed cache/pilot.pt \
  --iters 3 \
  --out runs/pilot \
  --writer ncm_iwsp/writers/stub.py
```

Do not point `--jsonl` at `pretrain_t2t.jsonl`, `sft_*.jsonl`, `dpo.jsonl`, `rlaif.jsonl`, or `agent_rl*.jsonl`. The packer refuses those names. Short `{"text": "..."}` paragraphs are hash-split **by document** into `visible_train`, `visible`, and `hidden`, then each split is packed into length-`L` windows. EOS is appended after each document. Only the tail of a stream is padded. The cache is a torch dict (`visible_train`, `visible`, `hidden`, `meta`), in the spirit of `packed_windows.pt`.

## What the loop does

Each iteration rebuilds both modules from `seed`, calls `write_weights`, audits the writer file, evaluates the **visible** windows, and appends a row to `results.csv`. The agent-facing scalar is `mean_level_ce`: the unweighted mean of per-level cross-entropies on levels 1..K-1 (level 0 is not a generator target). `length_weighted_ce` is the official alpha=0 weighting, where every code position counts equally.

Also logged, on the visible split unless noted:

- tokenizer reconstruction CE and accuracy, PAD ignored
- codebook usage and dead-code fraction on this batch (a diagnostic, not an EMA update)
- per-level CE and exact-code accuracy
- reconstruction accuracy after keeping gold level 0 and inserting the argmax of the written levels
- baselines written **once**, before the writer runs: random init, most-frequent-token accuracy, add-one token unigram NLL, always-PAD constant CE, `log(codebook)` for a uniform finer level, level-0 unigram NLL, and mean add-one NLL of the predicted levels
- a few qualitative decodes: level-0 unigram plus random finer codes, then level-0 unigram plus the generator, plus three fixed prompts

`most_frequent_token_ce` / `always_pad_ce` use logits of +20 on the constant class and −20 elsewhere so the number is finite. Compare models with `mean_level_ce`, `recon_ce`, `token_unigram_nll`, and `mean_predicted_level_unigram_nll` instead. Those are ordinary cross-entropies. On the smoke codebook, a uniform finer level sits at `log(64) ≈ 4.159`.

Hidden-split CE is written only to `hidden_metrics.jsonl`. `results.csv` has no hidden column. Do not paste the hidden file into an agent prompt.

The shipped writer is `ncm_iwsp/writers/stub.py`. Iteration 0 writes nothing. Later iterations write a sinusoid into the token embedding, axis-aligned codebook rows, and a scaled identity into the generator input projection. That is a closed form, not a result.

## Plugging in a later Claude or Opus loop

This repository does not call a paid API. A later session can follow the csinva `launch_claude.py` shape without pointing it at NCM-GPT1:

1. Copy `ncm_iwsp/writers/stub.py` to a run directory. The dry loop already snapshots the writer next to the metrics.
2. The agent may edit only that file: `write_weights`, `MODEL_SHORTHAND`, and `MODEL_DESCRIPTION` (at least 80 characters, saying what the codebooks and embeddings are).
3. The agent must not edit `ncm_iwsp/evaluate.py`, `ncm_iwsp/audit.py`, or `ncm_iwsp/packing.py`, and must not read `hidden_metrics.jsonl`.
4. Show the agent one number from `results.csv`: `mean_level_ce`.
5. Keep `torch.manual_seed` at the start of `write_weights`. Two calls with the same iteration must produce the same tensors. The audit checks that.
6. Write code that **generates** tensors (loops, closed forms, small tables). Do not paste millions of floats, and do not fit those tables with a gradient step.
7. A 200-word describe-and-rebuild audit, as in the iWSP paper, is a later human-run session. It is not automated here.

## What success is

On the **hidden** split, after the audit passes: tokenizer reconstruction and/or `mean_level_ce` beats **both** random init and the evaluator’s unigram baseline by a gap chosen before looking at the hidden numbers. The description has to explain the weights (named circuits, closed-form tables), not a dump of floats.

Thirty audited iterations with no training violations is the harness target from the study. Beating MiniMind-3, official NCM perplexity, or NCM-GPT1’s BookCorpus validation CE is not the target. A clean failure is a valid result: both weight-space papers say deep residual representations are not yet something these writers learn.

成功的標準是：通過審計，而且在隱藏集上明顯勝過隨機初始化和 evaluator 裡的 unigram。不是追上 MiniMind-3，也不是 NCM-GPT1 的驗證 CE。失敗本身可以是結果。

## Tests

```bash
pytest
```

Covers packing (shared windows, document split, refusal of SFT/full-pretrain names), both pretoks, the parameter counts above, evaluator metrics and PAD masking, audit rejections, a smoke and pilot forward, and a two-iteration dry loop. GitHub Actions runs the same suite on CPU PyTorch.

## Layout

| Path | Role |
| --- | --- |
| `configs/smoke.yaml` | Rung 0 |
| `configs/pilot.yaml` | Rung 1 |
| `ncm_iwsp/pretok.py` | Frozen ascii / byte pretok |
| `ncm_iwsp/packing.py` | Document split and window packing |
| `ncm_iwsp/models/` | Inference-only VQ-VAE and generator |
| `ncm_iwsp/writers/stub.py` | The only file a writer should edit |
| `ncm_iwsp/evaluate.py` | Fixed metrics |
| `ncm_iwsp/audit.py` | Training, download, and hidden-split checks |
| `ncm_iwsp/loop.py` | Propose / write / eval / audit |
| `ncm_iwsp/cli.py` | `pack`, `count`, `eval`, `loop` |

## Leftover risks

- There is no published evidence that iWSP can program a deep residual code hierarchy. The papers call that an open problem. This harness can record a negative.
- The audit is a source scan plus a determinism check. It will miss a training step hidden behind an imported helper whose source it does not read.
- Smoke ASCII throws Chinese away. Pilot bytes keep it, at three tokens per CJK character, so a length-64 window holds about 21 Chinese characters before packing overhead. That ratio is the pretok’s definition, not a measurement of `pretrain_t2t_mini.jsonl` (that file was not downloaded here).
- Level-0 and per-level unigrams are add-one counts on the frozen visible-train windows of the **current** codebooks. They are baselines, not weights. A tiny visible set makes them noisy.
- Qualitative samples at random init are not language. Prefix length is 1, so a “prompt” is a single token plus a unigram level 0.
- The real 1.2GB jsonl has not been packed in this environment. `data.max_windows` caps how many windows an iteration scores; raise it only when a CPU or a **new** GPU budget is acceptable.
- Do not use the NCM-GPT1 Kaggle T4×2 quota for this search unless Nicholas explicitly reallocates it.
