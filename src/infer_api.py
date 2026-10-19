"""Run a flagship API model on a TAT-QA split with the same contract, few-shot examples and scorer.

Fairness: the API model gets the output contract, 4 worked examples and is allowed to reason before
answering (plus whatever built-in thinking the model does). Its equation is executed by the same Python
calculator as the SLM's. Requests are resumable: re-running the same --run skips finished questions.

    # Claude (ANTHROPIC_API_KEY)
    python src/infer_api.py --provider anthropic --model claude-opus-5-5 --run claude_opus_5_5 --limit 500
    # OpenAI (OPENAI_API_KEY) -- pass that model's current $/1M token prices
    python src/infer_api.py --provider openai --model <model-id> --run openai_flagship --price_in X --price_out Y
    # Gemini via its OpenAI-compatible endpoint (GEMINI_API_KEY)
    python src/infer_api.py --provider gemini --model <model-id> --run gemini_flagship --price_in X --price_out Y
    # Any OpenAI-compatible host (OpenRouter, Together, Groq, ...) e.g. for Llama-70B-class open models
    python src/infer_api.py --provider openai_compatible --base_url https://openrouter.ai/api/v1 \
        --api_key_env OPENROUTER_API_KEY --model <model-id> --run llama70b --price_in X --price_out Y
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_io import latency_stats, load_few_shot, load_split, make_record, messages_for, write_run  # noqa: E402

# USD per 1M tokens (input, output), Anthropic first-party list prices as of Oct 2026.
# Other providers change prices often -- pass --price_in/--price_out from their pricing page.
KNOWN_PRICES = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
OPENAI_COMPAT_URLS = {
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai", "gemini", "openai_compatible"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a fixed subset to cap API spend")
    ap.add_argument("--few_shot", default="data/few_shot.json")
    ap.add_argument("--preds_dir", default="results/preds")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=16000, help="room for built-in thinking + answer")
    ap.add_argument("--effort", default=None,
                    help="reasoning effort (Claude: low|medium|high|xhigh|max; OpenAI-style: reasoning_effort). "
                         "Default: the provider's default")
    ap.add_argument("--price_in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price_out", type=float, default=None, help="USD per 1M output tokens (incl. thinking)")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key_env", default=None)
    ap.add_argument("--rate_limit_wait", type=float, default=600,
                    help="seconds to keep retrying a request rejected with 429 / 'at capacity' before recording it "
                         "as failed (free tiers on busy models need patience)")
    ap.add_argument("--max_minutes", type=float, default=None,
                    help="stop starting new requests after this many minutes; unfinished questions are retried "
                         "on the next run (keeps a Kaggle session from running out of time)")
    return ap.parse_args()


class AnthropicClient:
    def __init__(self, args):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=6)
        self.args = args

    def __call__(self, messages):
        system, turns = messages[0]["content"], messages[1:]
        kwargs = {}
        if self.args.effort:
            kwargs["output_config"] = {"effort": self.args.effort}
        # No server-side fallback on purpose: every answer must come from the model being benchmarked.
        resp = self.client.messages.create(model=self.args.model, max_tokens=self.args.max_tokens,
                                           system=system, messages=turns, **kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError("refusal")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens, resp.stop_reason


class OpenAICompatClient:
    def __init__(self, args):
        from openai import OpenAI
        base_url, key_env = OPENAI_COMPAT_URLS.get(args.provider, (args.base_url, args.api_key_env))
        base_url = args.base_url or base_url
        key_env = args.api_key_env or key_env
        if not key_env or key_env not in os.environ:
            sys.exit(f"set the API key environment variable ({key_env or '--api_key_env'})")
        self.client = OpenAI(base_url=base_url, api_key=os.environ[key_env], max_retries=6)
        self.args = args

    def __call__(self, messages):
        kwargs = {"reasoning_effort": self.args.effort} if self.args.effort else {}
        # OpenAI's reasoning models only take max_completion_tokens; other compatible hosts expect max_tokens.
        limit_key = "max_completion_tokens" if self.args.provider == "openai" else "max_tokens"
        kwargs[limit_key] = self.args.max_tokens
        resp = self.client.chat.completions.create(model=self.args.model, messages=messages, **kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return (choice.message.content or "", getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0), choice.finish_reason)


def is_rate_limited(e: Exception) -> bool:
    return type(e).__name__ == "RateLimitError" or getattr(e, "status_code", None) == 429


class OutOfTime(Exception):
    pass


def call_with_patience(call, messages, max_wait: float, deadline: float = float("inf")):
    """Retry 429s with growing pauses (on top of the SDK's own quick retries), never past `deadline`.

    Returns (result, seconds the successful attempt took) so waiting never counts as model latency."""
    waited, delay = 0.0, 15.0
    while True:
        if time.time() >= deadline:
            raise OutOfTime("time limit reached - re-run to continue")
        t = time.perf_counter()
        try:
            return call(messages), time.perf_counter() - t
        except Exception as e:
            if not is_rate_limited(e) or waited >= max_wait:
                raise
            pause = min(delay, max_wait - waited, max(0.0, deadline - time.time())) * random.uniform(0.8, 1.0)
            time.sleep(pause)
            waited += pause
            delay = min(delay * 2, 120.0)


def main():
    args = parse_args()
    price_in, price_out = args.price_in, args.price_out
    if price_in is None and args.model in KNOWN_PRICES:
        price_in, price_out = KNOWN_PRICES[args.model]
    if price_in is None or price_out is None:
        print("WARNING: no prices given (--price_in/--price_out); cost will be reported as unknown")

    rows = load_split(args.data, args.split, args.limit)
    few_shot = load_few_shot(args.few_shot)
    call = AnthropicClient(args) if args.provider == "anthropic" else OpenAICompatClient(args)

    preds_dir = Path(args.preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    partial = preds_dir / f"{args.run}.partial.jsonl"
    done = {}
    if partial.exists():
        for line in open(partial, encoding="utf-8"):
            r = json.loads(line)
            if not str(r.get("error") or "").startswith("api_error"):
                done[r["uid"]] = r
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{len(rows)} questions, {len(done)} already done, {len(todo)} to run")

    lock = threading.Lock()
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else float("inf")

    def work(row):
        try:
            (text, tin, tout, stop), lat = call_with_patience(
                call, messages_for(row, "fewshot", few_shot), args.rate_limit_wait, deadline)
        except Exception as e:  # recorded and retried on the next run
            return make_record(row, None, error=f"api_error: {type(e).__name__}: {str(e)[:200]}")
        cost = None if price_in is None else (tin * price_in + tout * price_out) / 1e6
        return make_record(row, text, in_tokens=tin, out_tokens=tout, latency_s=lat,
                           cost_usd=cost, stop_reason=stop)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool, open(partial, "a", encoding="utf-8") as fh:
        futures = [pool.submit(work, r) for r in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["uid"]] = rec
            if k % 25 == 0 or k == len(futures):
                print(f"  {k}/{len(futures)}")
    wall = time.perf_counter() - t0

    records = [done[r["uid"]] for r in rows if r["uid"] in done]
    ok = [r for r in records if r.get("cost_usd") is not None]
    api_errors = sum(1 for r in records if str(r.get("error") or "").startswith("api_error"))
    mean_cost = sum(r["cost_usd"] for r in ok) / len(ok) if ok else None
    meta = {
        "kind": "api", "provider": args.provider, "model": args.model, "mode": "fewshot",
        "split": args.split, "effort": args.effort, "concurrency": args.concurrency,
        "wall_time_s_this_session": wall, "api_errors": api_errors,
        "in_tokens": sum(r.get("in_tokens") or 0 for r in records),
        "out_tokens": sum(r.get("out_tokens") or 0 for r in records),
        "price_in_per_1m": price_in, "price_out_per_1m": price_out,
        "cost_usd_total": sum(r["cost_usd"] for r in ok) if ok else None,
        "cost_per_1m_questions_usd": mean_cost * 1e6 if mean_cost is not None else None,
        **latency_stats([r.get("latency_s") for r in records]),
    }
    write_run(args.preds_dir, args.run, records, meta)
    if api_errors:
        first = next(r["error"] for r in records if str(r.get("error") or "").startswith("api_error"))
        print(f"{api_errors} requests failed; re-run the same command to retry them. First error:\n  {first}")
    print({k: meta[k] for k in ("cost_usd_total", "cost_per_1m_questions_usd", "latency_p50_s")})


if __name__ == "__main__":
    main()
