"""Download TAT-QA and build train/val/test JSONL files plus a fixed few-shot set.

Splits: official train -> train, official dev -> val, official test (gold released) -> test.

    python src/prepare_data.py --out data
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tatqa_common import build_target, build_user_prompt, format_context, target_to_text  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/"
FILES = {
    "train": "dataset_raw/tatqa_dataset_train.json",
    "val": "dataset_raw/tatqa_dataset_dev.json",
    "test": "dataset_raw/tatqa_dataset_test_gold.json",
}
OFFICIAL_EVAL = ["tatqa_eval.py", "tatqa_metric.py", "tatqa_utils.py"]


def download(rel: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {rel}")
        urllib.request.urlretrieve(BASE_URL + rel, dest)
    return dest


def build_rows(items: list[dict], split: str) -> tuple[list[dict], collections.Counter]:
    rows, stats = [], collections.Counter()
    for item in items:
        context = format_context(item)
        for q in item["questions"]:
            target = build_target(q)
            stats["total"] += 1
            if target is None:
                stats[f"no_target_{q['answer_type']}"] += 1
                if split == "train":
                    continue  # never train on a target we could not verify
            rows.append({
                "uid": q["uid"],
                "split": split,
                "question": q["question"],
                "user_prompt": build_user_prompt(context, q["question"]),
                "target": target_to_text(target) if target else None,
                "answer_type": q["answer_type"],
                "answer_from": q["answer_from"],
                "scale": q["scale"],
                "gold_answer": q["answer"],
            })
    stats["kept"] = len(rows)
    return rows, stats


def pick_few_shot(items: list[dict]) -> list[dict]:
    """One short, verified example per answer type, chosen deterministically from train."""
    wanted = ["span", "multi-span", "arithmetic", "count"]
    best: dict[str, tuple[int, dict]] = {}
    for item in items:
        context = format_context(item)
        for q in item["questions"]:
            t = q["answer_type"]
            target = build_target(q)
            if t not in wanted or target is None:
                continue
            if t == "arithmetic" and q["scale"] != "percent":
                continue  # show the percentage convention, the most common trap
            if t == "span" and not q["scale"]:
                continue  # show a scaled number
            size = len(context)
            if t not in best or size < best[t][0]:
                best[t] = (size, {
                    "uid": q["uid"],
                    "user": build_user_prompt(context, q["question"]),
                    "assistant": "```json\n" + target_to_text(target) + "\n```",
                })
    return [best[t][1] for t in wanted if t in best]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in OFFICIAL_EVAL:
        download(f, raw / f)

    train_items = None
    for split, rel in FILES.items():
        path = download(rel, raw / Path(rel).name)
        items = json.loads(path.read_text(encoding="utf-8"))
        if split == "train":
            train_items = items
        rows, stats = build_rows(items, split)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, dict(stats))

    few_shot = pick_few_shot(train_items)
    (out / "few_shot.json").write_text(json.dumps(few_shot, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"few-shot examples: {[fs['uid'] for fs in few_shot]}")
    # The official gold file used for scoring the test split:
    print(f"gold file: {raw / Path(FILES['test']).name}")


if __name__ == "__main__":
    main()
