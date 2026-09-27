"""Score distil labs prediction files: exact JSON answer match, unanswered rows, and reasoning length in tokens."""

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen3.5-4B"


def parse_answer(text: str | None) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def as_dict(value) -> dict:
    return ast.literal_eval(value) if isinstance(value, str) else value


def score(path: Path, tokenizer) -> str:
    rows = [json.loads(line) for line in path.open()]
    correct, unanswered, tokens = 0, 0, []
    for row in rows:
        gold = parse_answer(as_dict(row["completion"])["content"])
        message = as_dict(row["prediction"])
        prediction = parse_answer(message.get("content"))
        reasoning = message.get("reasoning_content") or ""
        if prediction is None:
            unanswered += 1
        else:
            correct += prediction == gold
            tokens.append(len(tokenizer.encode(reasoning)))
    median = statistics.median(tokens) if tokens else 0
    return f"{path}: {correct}/{len(rows)} correct, {unanswered} unanswered, median reasoning tokens (answered rows) {median}"


def main(paths: list[Path]):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for path in paths:
        print(score(path, tokenizer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    main(paths=args.paths)
