"""Score every run with the official TAT-QA metric and build the SLM-vs-LLM comparison.

Two views are reported:
  * full:   each run on every question it answered
  * common: every run on the questions ALL runs answered (use this when API runs used --limit)

    python src/evaluate.py --preds_dir results/preds --out results --hero slm_finetuned
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
TYPES = ["span", "multi-span", "arithmetic", "count"]


def load_official_metric(raw_dir: Path):
    sys.path.insert(0, str(raw_dir))
    from tatqa_metric import TaTQAEmAndF1  # official scorer, downloaded by prepare_data.py
    return TaTQAEmAndF1


def load_gold(path: Path) -> dict:
    gold = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        for q in item["questions"]:
            gold[q["uid"]] = q
    return gold


def load_runs(preds_dir: Path) -> dict:
    runs = {}
    for meta_path in sorted(preds_dir.glob("*.meta.json")):
        name = meta_path.name[: -len(".meta.json")]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recs = [json.loads(l) for l in open(preds_dir / f"{name}.jsonl", encoding="utf-8")]
        if not recs:  # e.g. an API model that answered nothing yet - it would empty the common view
            print(f"skipping {name}: no answers yet")
            continue
        runs[name] = {"meta": meta, "recs": {r["uid"]: r for r in recs}}
    return runs


def was_cut_off(rec: dict | None, meta: dict) -> bool:
    """The answer stopped because it hit the output-token limit (thinking + answer), not because it was done."""
    if not rec:
        return False
    if rec.get("stop_reason") in ("length", "max_tokens"):  # OpenAI-style / Anthropic API
        return True
    if meta.get("kind") == "local" and rec.get("out_tokens") is not None:
        limit = meta.get("max_new_tokens") or (256 if meta.get("mode") == "finetuned" else 1024)
        return rec["out_tokens"] >= limit
    return False


def score(metric_cls, gold: dict, recs: dict, uids: list[str], meta: dict | None = None) -> dict:
    m = metric_cls()
    for uid in uids:
        r = recs.get(uid)
        answer, scale = (r["answer"], r["scale"]) if r else (None, "")
        m(ground_truth=gold[uid], prediction=answer, pred_scale=scale or "")
    em, f1, scale_acc, _ = m.get_overall_metric()
    out = {"n": len(uids), "em": em * 100, "f1": f1 * 100, "scale_acc": scale_acc * 100}
    details = m.get_raw()
    for t in TYPES:
        xs = [d["em"] for d in details if d["answer_type"] == t]
        out[f"em_{t}"] = 100 * sum(xs) / len(xs) if xs else None
    out["unparsable_pct"] = 100 * sum(1 for u in uids if recs.get(u) is None or recs[u]["error"]) / max(len(uids), 1)
    out["cut_off_pct"] = 100 * sum(1 for u in uids if was_cut_off(recs.get(u), meta or {})) / max(len(uids), 1)
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def money(x):
    if x is None:
        return "-"
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def markdown_table(rows: list[dict], hero: str | None) -> str:
    head = ("| Run | Model | n | EM | F1 | EM span | EM multi-span | EM arithmetic | EM count | "
            "Unparsable % | Cut off % | Latency p50 (s) | $ per 1M questions |")
    lines = [head, "|" + "---|" * 13]
    for r in rows:
        name = f"**{r['run']}**" if r["run"] == hero else r["run"]
        lines.append(" | ".join([
            f"| {name}", r["model"], str(r["n"]), fmt(r["em"]), fmt(r["f1"]), fmt(r["em_span"]),
            fmt(r["em_multi-span"]), fmt(r["em_arithmetic"]), fmt(r["em_count"]), fmt(r["unparsable_pct"]),
            fmt(r["cut_off_pct"]),
            fmt(r["latency_p50_s"], 2), money(r["cost_per_1m_questions_usd"])]) + " |")
    return "\n".join(lines)


def charts(rows: list[dict], hero: str | None, out: Path, title_suffix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accent, muted, ink, grid = "#2563eb", "#9ca3af", "#1f2937", "#e5e7eb"
    rows = sorted(rows, key=lambda r: r["em"])

    # 1) Exact match per run
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4))
    colors = [accent if r["run"] == hero else muted for r in rows]
    ax.barh([r["run"] for r in rows], [r["em"] for r in rows], color=colors, height=0.6)
    for i, r in enumerate(rows):
        ax.text(r["em"] + 0.8, i, f"{r['em']:.1f}", va="center", color=ink, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exact match (%) - official TAT-QA metric")
    ax.set_title(f"TAT-QA exact match{title_suffix}", loc="left", color=ink)
    ax.grid(axis="x", color=grid)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=160)
    plt.close(fig)

    # 2) Accuracy vs cost (log x): the "is the big model worth it?" picture
    priced = [r for r in rows if r["cost_per_1m_questions_usd"]]
    if len(priced) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in priced:
            is_hero = r["run"] == hero
            ax.scatter(r["cost_per_1m_questions_usd"], r["em"], s=90 if is_hero else 60,
                       color=accent if is_hero else muted, edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(r["run"], (r["cost_per_1m_questions_usd"], r["em"]), textcoords="offset points",
                        xytext=(8, 4), fontsize=9, color=ink)
        ax.set_xscale("log")
        ax.margins(x=0.2, y=0.1)
        ax.set_xlabel("Cost per 1M questions (USD, log scale)")
        ax.set_ylabel("Exact match (%)")
        ax.set_title(f"Accuracy vs cost{title_suffix}", loc="left", color=ink)
        ax.grid(color=grid)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "accuracy_vs_cost.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="results/preds")
    ap.add_argument("--gold", default="data/raw/tatqa_dataset_test_gold.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--hero", default="slm_finetuned", help="run name to highlight")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metric_cls = load_official_metric(Path(args.gold).parent)
    gold = load_gold(Path(args.gold))
    runs = load_runs(Path(args.preds_dir))
    if not runs:
        sys.exit(f"no runs in {args.preds_dir}")

    common = set(gold)
    for run in runs.values():
        common &= set(run["recs"])
    common = sorted(common)

    report = ["# SLM vs flagship LLMs on TAT-QA (financial report QA)\n"]
    for view, title_suffix in (("full", ""), ("common", " (same questions for every run)")):
        rows = []
        for name, run in runs.items():
            uids = sorted(u for u in run["recs"] if u in gold) if view == "full" else common
            meta = run["meta"]
            rows.append({"run": name, "model": meta.get("model"), "kind": meta.get("kind"),
                         **score(metric_cls, gold, run["recs"], uids, run["meta"]),
                         "latency_p50_s": meta.get("latency_p50_s"),
                         "cost_per_1m_questions_usd": meta.get("cost_per_1m_questions_usd")})
        rows.sort(key=lambda r: -r["em"])
        with open(out / f"summary_{view}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        report += [f"## {view.capitalize()} view{title_suffix}\n", markdown_table(rows, args.hero), ""]
        if view == "common" or len(runs) == 1 or len(common) == len(gold):
            charts(rows, args.hero, out, title_suffix if view == "common" else "")

    report += [
        "Notes:",
        "- EM/F1: official TAT-QA scorer (numbers rounded to 2 decimals and multiplied by scale; scale must match).",
        "- Arithmetic answers are computed by executing each model's equation in Python, for every model.",
        "- Local cost = GPU $/hour x measured batch wall time; API cost = measured tokens x list price "
        "(thinking tokens included).",
        "- Latency p50 = single request at a time (local) / per request under concurrency (API).",
        "- Cut off % = answers that hit the output-token limit (thinking + answer) before finishing; "
        "these count as wrong, so a high value means the limit, not the model, cost points.",
    ]
    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {out / 'summary.md'}, summary_*.csv and charts")


if __name__ == "__main__":
    main()
