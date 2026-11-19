"""Command line for packing, counting, evaluation, and the local dry loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ncm_iwsp.audit import AuditError
from ncm_iwsp.build import build_models, parameter_report
from ncm_iwsp.config import load_config
from ncm_iwsp.evaluate import evaluate_windows
from ncm_iwsp.loop import load_writer, run_dry_loop
from ncm_iwsp.packing import demo_texts, iter_jsonl_texts, load_cache, make_cache, save_cache
from ncm_iwsp.pretok import CompactPretok


def _cmd_pack(args) -> int:
    cfg = load_config(args.config)
    pretok = CompactPretok.from_config(cfg)
    if args.demo:
        texts = demo_texts()
    else:
        if not args.jsonl:
            raise SystemExit("pack needs --jsonl PATH or --demo")
        texts = list(iter_jsonl_texts(args.jsonl))
    data = cfg.get("data", {})
    hidden_pct = int(data.get("hidden_pct", 10))
    visible_pct = int(data.get("visible_pct", 10))
    # The built-in paragraph list is only a few dozen rows. A 1% hash bucket
    # would often be empty, so the demo forces a coarser split than a real corpus.
    if args.demo:
        hidden_pct = max(hidden_pct, 10)
        visible_pct = max(visible_pct, 10)
    cache = make_cache(
        texts,
        pretok,
        context_length=int(cfg["tokenizer"]["context_length"]),
        hidden_pct=hidden_pct,
        visible_pct=visible_pct,
        max_windows=int(data.get("max_windows", 0) or 0),
    )
    save_cache(cache, args.out)
    print(json.dumps(cache["meta"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


def _cmd_count(args) -> int:
    cfg = load_config(args.config)
    _, vqvae, generator = build_models(cfg)
    report = parameter_report(vqvae, generator)
    report["name"] = cfg["name"]
    print(json.dumps(report, indent=2))
    return 0


def _cmd_eval(args) -> int:
    cfg = load_config(args.config)
    cache = load_cache(args.packed)
    writer = load_writer(args.writer)
    pretok, vqvae, generator = build_models(cfg)
    writer.write_weights(vqvae, generator, iteration=int(args.iteration))
    split = cache["visible"] if args.split == "visible" else cache["visible_train"]
    metrics = evaluate_windows(
        pretok,
        vqvae,
        generator,
        split,
        train_windows=cache["visible_train"],
        batch_size=int(cfg.get("eval", {}).get("batch_size", 4)),
        max_windows=int(cfg.get("eval", {}).get("max_windows", 32)),
        n_generate=int(cfg.get("eval", {}).get("n_generate", 2)),
        temperature=0.0,
    )
    shown = {
        "split": args.split,
        "iteration": int(args.iteration),
        "scalar": metrics["scalar"],
        "mean_level_ce": metrics["mean_level_ce"],
        "length_weighted_ce": metrics["length_weighted_ce"],
        "recon_acc": metrics["recon_acc"],
        "recon_ce": metrics["recon_ce"],
        "mean_exact_acc": metrics["mean_exact_acc"],
        "recon_after_written_levels_acc": metrics["recon_after_written_levels_acc"],
        "codebook_mean_usage": metrics["codebook_usage"]["mean_usage"],
        "most_frequent_token_ce": metrics["baselines"]["most_frequent_token_ce"],
        "most_frequent_token_acc": metrics["baselines"]["most_frequent_token_acc"],
        "token_unigram_nll": metrics["baselines"]["token_unigram_nll"],
        "always_pad_ce": metrics["baselines"]["always_pad_ce"],
        "level0_unigram_nll": metrics["baselines"]["level0_unigram_nll"],
        "mean_predicted_level_unigram_nll": metrics["baselines"]["mean_predicted_level_unigram_nll"],
        "uniform_finer_ce": metrics["baselines"]["uniform_finer_ce"],
        "qualitative": metrics["baselines"]["qualitative"],
        "description": getattr(writer, "LAST_DESCRIPTION", ""),
    }
    print(json.dumps(shown, indent=2, ensure_ascii=False))
    return 0


def _cmd_loop(args) -> int:
    cfg = load_config(args.config)
    cache = load_cache(args.packed)
    try:
        summary = run_dry_loop(cfg, cache, args.writer, args.out, iterations=int(args.iters))
    except AuditError as exc:
        print("audit rejected the writer:")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 2
    print(json.dumps(summary, indent=2))
    print(
        "Agent-facing scalar is mean_level_ce in results.csv. "
        f"Hidden numbers are only in {summary['hidden_log']} and must not be shown to the writer."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncm_iwsp",
        description="NCM-MiniMind-iWSP pilot harness. Does not train and is not NCM-GPT1.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pack = sub.add_parser("pack", help="pack a jsonl, or the built-in demo paragraphs")
    pack.add_argument("--config", required=True)
    pack.add_argument("--jsonl", default=None, help="JSONL of {\"text\": ...}. Do not commit the MiniMind file.")
    pack.add_argument("--demo", action="store_true", help="pack the synthetic paragraphs shipped with the code")
    pack.add_argument("--out", required=True)
    pack.set_defaults(func=_cmd_pack)

    count = sub.add_parser("count", help="instantiate a config and print parameter counts")
    count.add_argument("--config", required=True)
    count.set_defaults(func=_cmd_count)

    evaluate = sub.add_parser("eval", help="evaluate a writer iteration on the visible split")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--packed", required=True)
    evaluate.add_argument("--writer", default="ncm_iwsp/writers/stub.py")
    evaluate.add_argument("--iteration", type=int, default=0)
    evaluate.add_argument("--split", choices=("visible", "visible_train"), default="visible")
    evaluate.set_defaults(func=_cmd_eval)

    loop = sub.add_parser("loop", help="local propose/write/eval/audit dry loop")
    loop.add_argument("--config", required=True)
    loop.add_argument("--packed", required=True)
    loop.add_argument("--writer", default="ncm_iwsp/writers/stub.py")
    loop.add_argument("--out", required=True)
    loop.add_argument("--iters", type=int, default=3)
    loop.set_defaults(func=_cmd_loop)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
