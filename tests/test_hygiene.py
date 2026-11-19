from pathlib import Path


def test_repo_does_not_contain_minimind_corpus_or_gpt1_checkout():
    root = Path(".")
    banned_names = {
        "pretrain_t2t_mini.jsonl",
        "pretrain_t2t.jsonl",
        "sft_t2t.jsonl",
        "sft_t2t_mini.jsonl",
        "dpo.jsonl",
        "rlaif.jsonl",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.parts[0] in {".git", ".venv", "venv"}:
            continue
        assert path.name not in banned_names
        assert "ncm-gpt1" not in path.parts
