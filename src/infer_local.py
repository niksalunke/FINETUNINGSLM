"""Run a local model (base or fine-tuned SLM, or any open model that fits) on a TAT-QA split.

Measures two things separately:
  * throughput: the whole split in one batched pass (how a production batch job would run)
  * latency:    --latency_n requests one at a time (what a single user waits)

    # fine-tuned SLM (merged), trained prompt format
    python src/infer_local.py --model outputs/qwen3-4b-tatqa-merged --run slm_finetuned --mode finetuned
    # same base model before fine-tuning, given few-shot examples + reasoning (its best shot)
    python src/infer_local.py --model Qwen/Qwen3-4B-Instruct-2507 --run slm_base_fewshot --mode fewshot
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_io import latency_stats, load_few_shot, load_split, make_record, messages_for, write_run  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="name for this run, e.g. slm_finetuned")
    ap.add_argument("--mode", choices=["finetuned", "fewshot"], default="finetuned")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--few_shot", default="data/few_shot.json")
    ap.add_argument("--preds_dir", default="results/preds")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="default: 256 finetuned, 1024 fewshot")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size (GPUs)")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--hf_batch_size", type=int, default=8)
    ap.add_argument("--latency_n", type=int, default=100, help="sequential single requests for latency")
    ap.add_argument("--gpu_hourly_usd", type=float, default=0.35,
                    help="what one of these GPUs costs per hour in the cloud (T4 ~0.35; check your provider)")
    return ap.parse_args()


class VLLMRunner:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=args.model, dtype="half", max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_mem, tensor_parallel_size=args.tp, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    def prompt(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, prompt):
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    def generate(self, prompts):
        outs = self.llm.generate(prompts, self.sp, use_tqdm=len(prompts) > 1)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]


class HFRunner:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        on_gpu = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16 if on_gpu else torch.float32,
            device_map={"": 0} if on_gpu else "cpu", attn_implementation="sdpa").eval()
        self.max_new = args.max_new_tokens
        self.bs = args.hf_batch_size

    def prompt(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, prompt):
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    def generate(self, prompts):
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        results = [None] * len(prompts)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for j, i in enumerate(idx):
                gen = out[j, enc["input_ids"].shape[1]:]
                n_out = int((gen != self.tok.pad_token_id).sum())
                n_in = int(enc["attention_mask"][j].sum())
                results[i] = (self.tok.decode(gen, skip_special_tokens=True), n_in, n_out)
            print(f"  {min(s + self.bs, len(order))}/{len(order)}", end="\r")
        return results


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 256 if args.mode == "finetuned" else 1024
    rows = load_split(args.data, args.split, args.limit)
    few_shot = load_few_shot(args.few_shot) if args.mode == "fewshot" else []

    runner = VLLMRunner(args) if args.backend == "vllm" else HFRunner(args)
    prompts = [runner.prompt(messages_for(r, args.mode, few_shot)) for r in rows]
    budget = args.max_model_len - args.max_new_tokens
    fits = [runner.n_tokens(p) <= budget for p in prompts]
    run_idx = [i for i, ok in enumerate(fits) if ok]
    print(f"{len(rows)} questions, {len(rows) - len(run_idx)} skipped as too long for max_model_len")

    runner.generate(prompts[:2])  # warm-up (CUDA graphs, kernels)

    # --- throughput: everything in one batched pass
    t0 = time.perf_counter()
    gens = runner.generate([prompts[i] for i in run_idx])
    wall = time.perf_counter() - t0
    by_idx = dict(zip(run_idx, gens))

    # --- latency: one request at a time
    lat_idx = run_idx[:: max(1, len(run_idx) // args.latency_n)][: args.latency_n]
    latency = {}
    for i in lat_idx:
        t = time.perf_counter()
        runner.generate([prompts[i]])
        latency[i] = time.perf_counter() - t

    records, n_in, n_out = [], 0, 0
    for i, row in enumerate(rows):
        if i not in by_idx:
            records.append(make_record(row, None, error="prompt_too_long"))
            continue
        text, ti, to = by_idx[i]
        n_in, n_out = n_in + ti, n_out + to
        records.append(make_record(row, text, in_tokens=ti, out_tokens=to, latency_s=latency.get(i)))

    n = len(run_idx)
    n_gpus = args.tp if args.backend == "vllm" else 1
    cost_per_q = args.gpu_hourly_usd * n_gpus * (wall / 3600) / max(n, 1)
    meta = {
        "kind": "local", "model": args.model, "mode": args.mode, "split": args.split,
        "backend": args.backend, "gpu": gpu_name(), "n_gpus": n_gpus,
        "wall_time_s": wall, "throughput_qps": n / wall if wall else None,
        "out_tokens_per_s": n_out / wall if wall else None,
        "in_tokens": n_in, "out_tokens": n_out,
        "gpu_hourly_usd": args.gpu_hourly_usd,
        "cost_usd_total": cost_per_q * n,
        "cost_per_1m_questions_usd": cost_per_q * 1e6,
        **latency_stats(list(latency.values())),
    }
    write_run(args.preds_dir, args.run, records, meta)
    print({k: meta[k] for k in ("throughput_qps", "latency_p50_s", "cost_per_1m_questions_usd")})


if __name__ == "__main__":
    main()
