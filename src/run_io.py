"""Reading inputs and writing predictions in one format shared by local and API runs.

Each run writes:
  <preds_dir>/<run>.jsonl      one row per question: uid, raw output, parsed answer/scale, error, tokens, latency, cost
  <preds_dir>/<run>.meta.json  run-level facts: model, mode, timings, throughput, cost assumptions
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from tatqa_common import build_messages, parse_output, to_prediction


def load_split(data_dir: str, split: str, limit: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8")]
    if limit and limit < len(rows):
        # deterministic stratified-ish sample: every k-th row keeps the answer-type mix
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def load_few_shot(path: str | None) -> list[dict]:
    if not path:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def messages_for(row: dict, mode: str, few_shot: list[dict]) -> list[dict]:
    """mode 'finetuned' = plain contract (what the SLM was trained on);
    mode 'fewshot'  = contract + worked examples + permission to reason (best shot for prompted models)."""
    if mode == "finetuned":
        return build_messages(row["user_prompt"])
    if mode == "fewshot":
        return build_messages(row["user_prompt"], few_shot=few_shot, reasoning=True)
    raise ValueError(mode)


def make_record(row: dict, raw: str | None, **extra) -> dict:
    answer, scale, error = to_prediction(parse_output(raw or ""))
    if raw is None and "error" in extra:
        error = extra.pop("error")
    else:
        extra.pop("error", None)
    return {"uid": row["uid"], "answer_type": row["answer_type"], "raw": raw,
            "answer": answer, "scale": scale, "error": error, **extra}


def percentile(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def latency_stats(latencies: list[float]) -> dict:
    lat = [x for x in latencies if x is not None]
    return {"latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
            "latency_mean_s": statistics.fmean(lat) if lat else None, "latency_n": len(lat)}


def write_run(preds_dir: str, run: str, records: list[dict], meta: dict) -> None:
    out = Path(preds_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{run}.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"run": run, "n": len(records), **meta}
    (out / f"{run}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    errs = sum(1 for r in records if r["error"])
    print(f"wrote {len(records)} predictions to {out / (run + '.jsonl')} ({errs} unparsable)")
