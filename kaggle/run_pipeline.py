"""Run the whole SLM-vs-LLM pipeline on Kaggle from the settings in kaggle/config.toml.

The notebook only pulls this repo, loads Kaggle Secrets into the environment and calls:
    !python kaggle/run_pipeline.py                       # stages from config.toml
    !python kaggle/run_pipeline.py --stages setup,data,dryrun
    !python kaggle/run_pipeline.py --dry                 # print the commands without running them

Every stage checks whether its work is already done, so re-running the notebook is always safe:
an interrupted training run continues from its last checkpoint and finished API answers are not paid for twice.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    sys.exit("Python 3.11+ is needed to read kaggle/config.toml")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
ALL_STAGES = ["setup", "data", "llm_smoketest", "dryrun", "speedtest", "train", "merge", "infer", "api", "evaluate"]
TRAIN_LIBS = ['transformers>=4.51', 'peft>=0.13', 'accelerate>=1.0', 'bitsandbytes>=0.43', "anthropic", "openai"]


class StopPipeline(Exception):
    """Raised to end the run cleanly (e.g. training paused at its time budget)."""


class Runner:
    def __init__(self, cfg: dict, dry: bool):
        self.cfg = cfg
        self.dry = dry
        self.t0 = time.time()
        self.repo = None  # private Hugging Face repo holding checkpoints + results (set by init_hub)

    # ------------------------------------------------------------------ Hugging Face backup
    def init_hub(self):
        """Find (or create) the private Hub repo that keeps checkpoints and results across Kaggle sessions."""
        t = self.cfg["train"]
        want = t.get("hub_repo", "")
        if not want:
            print("Hub backup: OFF (train.hub_repo is empty) - nothing survives the end of this session")
            return
        if not os.environ.get("HF_TOKEN"):
            print("Hub backup: OFF - the HF_TOKEN secret is not attached to this notebook")
            return
        name = t.get("hub_name", "qwen3-4b-tatqa")
        if self.dry:
            self.repo = want if want != "auto" else f"<hf-user>/{name}"
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo = f"{api.whoami()['name']}/{name}" if want == "auto" else want
            api.create_repo(repo, private=True, exist_ok=True)  # also proves the token can write
            self.repo = repo
            print(f"Hub backup: ON -> private repo https://huggingface.co/{repo}")
        except Exception as e:
            print(f"Hub backup: FAILED ({type(e).__name__}: {str(e)[:200]})")
            print("  Check that HF_TOKEN is a token with WRITE access (huggingface.co -> Settings -> Access Tokens).")

    def restore_results(self):
        """On a fresh machine, bring back answers saved by earlier sessions (incl. partial API runs)."""
        if not self.repo or self.dry:
            return
        if list((ROOT / "results/preds").glob("*.jsonl")):
            return  # this machine already has results
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(self.repo, allow_patterns=["results/*"], local_dir=ROOT)
            found = sorted(f.name for f in (ROOT / "results/preds").glob("*.meta.json"))
            print(f"restored earlier results from the Hub: {found or 'none yet'}")
        except Exception as e:
            print(f"could not restore results from the Hub ({type(e).__name__}) - continuing without them")

    def backup_results(self, label: str):
        if not self.repo or self.dry or not (ROOT / "results").is_dir():
            return
        try:
            from huggingface_hub import upload_folder
            upload_folder(repo_id=self.repo, folder_path=str(ROOT / "results"), path_in_repo="results",
                          commit_message=f"results after {label}")
            print(f"results backed up to the Hub ({label})")
        except Exception as e:
            print(f"WARNING: results backup failed ({type(e).__name__}: {str(e)[:150]})")

    def minutes_left(self, reserve_minutes: float = 0) -> float:
        """Minutes left in the Kaggle session (a commit is killed at 12 h, which can lose its outputs)."""
        session = self.cfg["pipeline"].get("session_hours", 11.5) * 60
        return session - (time.time() - self.t0) / 60 - reserve_minutes

    # ------------------------------------------------------------------ helpers
    def sh(self, cmd: list[str], check: bool = True) -> int:
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if self.dry:
            return 0
        code = subprocess.run(cmd, cwd=ROOT).returncode  # output streams straight into the notebook
        if code and check:
            raise RuntimeError(f"command failed with exit code {code}")
        return code

    @staticmethod
    def exists(p: str) -> bool:
        return (ROOT / p).exists()

    @staticmethod
    def n_gpus() -> int:
        if not shutil.which("nvidia-smi"):
            return 0
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))

    def need_gpu(self):
        if self.n_gpus() == 0 and not self.dry:
            raise RuntimeError("no GPU in this session - set Accelerator to 'GPU T4 x2' in the notebook settings")

    @property
    def train_done(self) -> bool:
        return self.exists(f"{self.cfg['train']['out']}/final_adapter/adapter_config.json")

    @property
    def hero_answered(self) -> bool:
        """The fine-tuned model has answered the test set (this session or an earlier one, via the Hub)."""
        return self.exists("results/preds/slm_finetuned.meta.json")

    @property
    def merged(self) -> str:
        return f"{self.cfg['train']['out']}-merged"

    def api_cmd(self, c: dict, limit: int, run: str, preds_dir: str) -> list[str]:
        cmd = [PY, "src/infer_api.py", "--provider", c["provider"], "--model", c["model"],
               "--run", run, "--limit", str(limit), "--preds_dir", preds_dir]
        if c["provider"] != "anthropic":
            if c.get("base_url"):
                cmd += ["--base_url", c["base_url"]]
            cmd += ["--api_key_env", c.get("key_secret", "LLM_API_KEY")]
        if "price_in" in c and "price_out" in c:
            cmd += ["--price_in", str(c["price_in"]), "--price_out", str(c["price_out"])]
        return cmd + shlex.split(c.get("extra", ""))

    def contestants(self) -> list[dict]:
        """Enabled [[api]] entries whose key secret is attached to this session."""
        ready = []
        for c in self.cfg.get("api", []):
            if not c.get("enabled", True):
                continue
            if not os.environ.get(c["key_secret"]) and not self.dry:
                print(f"{c['name']}: {c['key_secret']} secret not attached - skipped")
                continue
            ready.append(c)
        return ready

    def train_args(self, out: str, override: dict | None = None) -> list[str]:
        t = {**self.cfg["train"], **(override or {})}
        args = ["src/train.py", "--model", self.cfg["model"]["base"], "--data", "data", "--out", out,
                "--lr", str(t["lr"]), "--batch_size", str(t["batch_size"]), "--grad_accum", str(t["grad_accum"]),
                "--max_len", str(t["max_len"])]
        if not t.get("quantize_4bit", True):
            args.append("--no_4bit")
        if not t.get("gradient_checkpointing", True):
            args.append("--no_grad_ckpt")
        if t.get("short_prompt"):
            args.append("--short_prompt")
        return args

    # ------------------------------------------------------------------ stages
    def setup(self):
        self.sh([PY, "-m", "pip", "install", "-q", "-U", *TRAIN_LIBS])
        # Kaggle ships an old torchao that current PEFT rejects when it builds plain (non-4-bit) LoRA layers.
        # Nothing in this project uses torchao, so remove it instead of fighting its version.
        if subprocess.run([PY, "-m", "pip", "show", "torchao"], capture_output=True).returncode == 0:
            self.sh([PY, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
        if shutil.which("nvidia-smi"):  # absent in a CPU-only session (the API comparison run)
            self.sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
        print(f"GPUs visible: {self.n_gpus()}")

    def data(self):
        if self.exists("data/test.jsonl") and self.exists("data/raw/tatqa_metric.py"):
            print("data already prepared")
            return
        self.sh([PY, "src/prepare_data.py", "--out", "data"])

    def llm_smoketest(self):
        """3 questions per contestant with short waits: catches a wrong model id, URL or key in minutes.
        Separate folder, so these answers never mix into the real comparison."""
        (ROOT / "results").mkdir(exist_ok=True)
        jobs = {f"{c['name']}_smoketest": self.api_cmd(c, 3, f"{c['name']}_smoketest", "results/smoketest")
                + ["--rate_limit_wait", "30", "--max_minutes", "5", "--request_timeout", "90"]
                for c in self.contestants()}
        if jobs:
            self.run_parallel(jobs, gpus=False, poll_seconds=15)

    def launch(self, args: list[str]):
        """Run train.py on every visible GPU (DDP via accelerate) -- the dry run and the real run use the same path."""
        n = max(1, self.n_gpus())
        if n > 1:
            launcher = ([shutil.which("accelerate"), "launch"] if shutil.which("accelerate")
                        else [PY, "-m", "accelerate.commands.launch"])
            self.sh([*launcher, "--multi_gpu", "--num_processes", str(n), "--mixed_precision", "fp16", *args])
        else:
            self.sh([PY, *args])

    def dryrun(self):
        self.need_gpu()
        d = self.cfg["dryrun"]
        out = "outputs/dryrun"
        if not self.dry:
            shutil.rmtree(ROOT / out, ignore_errors=True)  # always a fresh test of the current code
        self.launch(self.train_args(out) + ["--max_train_samples", str(d["samples"]),
                                            "--val_samples", str(d["val_samples"]), "--epochs", "1",
                                            "--save_steps", "1000"])
        print(f"DRY RUN OK - training works on this machine ({max(1, self.n_gpus())} GPU)")
        self.estimate_training_time(ROOT / out / "train_metrics.json")

    def speedtest(self):
        """Short training run of each [[speedtest]] setup; prints speed, memory and the fastest setup that fits."""
        self.need_gpu()
        d = self.cfg["dryrun"]
        n_train = sum(1 for _ in open(ROOT / "data/train.jsonl", encoding="utf-8")) if self.exists("data/train.jsonl") else 0
        epochs, max_len = self.cfg["train"]["epochs"], self.cfg["train"]["max_len"]
        rows = []
        for v in self.cfg.get("speedtest", []):
            v = {"gradient_checkpointing": True, **v}
            out = f"outputs/speedtest/{v['name']}"
            print(f"\n----- {v['name']}: 4-bit={v['quantize_4bit']}  batch {v['batch_size']} x accum {v['grad_accum']}"
                  f"  checkpointing={v['gradient_checkpointing']}  short prompt={self.cfg['train'].get('short_prompt')}")
            override = {k: v[k] for k in ("quantize_4bit", "batch_size", "grad_accum", "gradient_checkpointing")}
            args = self.train_args(out, override) + [
                "--max_train_samples", str(d["samples"]), "--val_samples", str(d["val_samples"]),
                "--epochs", "1", "--save_steps", "1000"]
            stamp = ROOT / out / "run_args.json"
            same = stamp.exists() and json.loads(stamp.read_text()) == args
            if same and self.exists(f"{out}/train_metrics.json") and not self.dry:
                print("already measured with these exact settings in this session - reusing the result")
            else:
                if not self.dry:
                    shutil.rmtree(ROOT / out, ignore_errors=True)
                try:
                    self.launch(args)
                except RuntimeError:
                    rows.append({**v, "status": "FAILED - see the error above (out of memory or a package problem)"})
                    continue
                if not self.dry:
                    stamp.write_text(json.dumps(args))
            if self.dry:
                continue
            m = json.loads((ROOT / out / "train_metrics.json").read_text())
            load, peak, total = m["gpu_mem_after_load_gb"], m["gpu_mem_peak_gb"], m["gpu_mem_total_gb"]
            # activations grow ~linearly with length: scale the dry run's longest example up to max_len
            worst = load + (peak - load) * max_len / max(1, m["longest_example_tokens"])
            speed = m["train_samples_per_second"]
            rows.append({**v, "status": "ok", "speed": speed, "peak": peak, "worst": worst, "total": total,
                         "hours": epochs * n_train / speed / 3600, "fits": worst <= total - 1.5})
        if self.dry or not rows:
            return
        print(f"\n{'setup':<20} {'4-bit':<6} {'batch':<6} {'ckpt':<6} {'ex/s':>6} {'peak GB':>8} "
              f"{'worst@' + str(max_len):>11} {'est. hours':>11}  status")
        for r in rows:
            head = f"{r['name']:<20} {str(r['quantize_4bit']):<6} {r['batch_size']:<6} {str(r['gradient_checkpointing']):<6}"
            if r["status"] != "ok":
                print(f"{head} {'':>6} {'':>8} {'':>11} {'':>11}  {r['status']}")
                continue
            note = "fits" if r["fits"] else "too close to the memory limit for the longest examples"
            print(f"{head} {r['speed']:>6.2f} {r['peak']:>8.1f} {r['worst']:>11.1f} {r['hours']:>11.1f}  {note}")
        ok = [r for r in rows if r["status"] == "ok" and r["fits"]]
        if ok:
            best = max(ok, key=lambda r: r["speed"])
            print(f"\nfastest setup that fits: {best['name']}  (~{best['hours']:.1f} h for {epochs} epochs)")
            print(f"to use it, set in kaggle/config.toml [train]: quantize_4bit = {str(best['quantize_4bit']).lower()}, "
                  f"batch_size = {best['batch_size']}, grad_accum = {best['grad_accum']}, "
                  f"gradient_checkpointing = {str(best['gradient_checkpointing']).lower()}")

    def estimate_training_time(self, metrics_path: Path):
        if self.dry or not metrics_path.exists():
            return
        speed = json.loads(metrics_path.read_text())["train_samples_per_second"]
        n_train = sum(1 for _ in open(ROOT / "data/train.jsonl", encoding="utf-8"))
        t = self.cfg["train"]
        hours = t["epochs"] * n_train / speed / 3600
        print()
        print(f"measured speed: {speed:.2f} examples/s  ->  full training ({t['epochs']} epochs x {n_train} "
              f"examples) needs about {hours:.1f} h (an upper estimate: the dry run includes start-up time)")
        if hours <= t["max_hours"]:
            print(f"fits in one session (time budget {t['max_hours']} h)")
        else:
            sessions = math.ceil(hours / t["max_hours"])
            backup = "set" if t.get("hub_repo") else "NOT SET - set train.hub_repo in kaggle/config.toml first"
            print(f"needs about {sessions} sessions: training pauses at {t['max_hours']} h and continues on the next "
                  f"Save & Run All. That requires the Hub backup (hub_repo: {backup}).")

    def fetch_final_adapter(self) -> bool:
        """Training finished in an earlier session: get its final adapter from the Hub."""
        if not self.repo or self.dry:
            return False
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(self.repo, allow_patterns=["final_adapter/*"], local_dir=ROOT / self.cfg["train"]["out"])
        except Exception as e:
            print(f"could not check the Hub for a final adapter ({type(e).__name__})")
        return self.train_done

    def train(self):
        if self.train_done or self.hero_answered:
            print("training already finished")
            return
        if self.fetch_final_adapter():
            print(f"training finished in an earlier session - final adapter downloaded from {self.repo}")
            return
        self.need_gpu()
        t = self.cfg["train"]
        # stop early enough to leave time for merge, inference and the report in this session
        reserve = self.cfg["pipeline"].get("reserve_after_train_minutes", 75)
        hours = min(t["max_hours"], self.minutes_left(reserve) / 60)
        if hours < 0.25 and not self.dry:
            raise StopPipeline("not enough session time left to train - run the notebook again")
        print(f"training time budget this session: {hours:.1f} h")
        args = self.train_args(t["out"]) + ["--epochs", str(t["epochs"]), "--save_steps", str(t["save_steps"]),
                                             "--max_hours", f"{hours:.2f}", "--resume"]
        args += ["--eval_steps", str(t.get("eval_steps", t["save_steps"]))]
        if self.repo:
            args += ["--hub_repo", self.repo]
            print(f"every checkpoint is uploaded to {self.repo}; a re-run continues from the latest one")
        elif t.get("require_hub_backup") and not self.dry:
            raise RuntimeError("no working Hub backup (see 'Hub backup' at the top of this log). Training is "
                               "refused because a stopped session would lose all progress. Fix the HF_TOKEN "
                               "secret, or set require_hub_backup = false in kaggle/config.toml")
        self.launch(args)
        if not self.train_done and not self.dry:
            raise StopPipeline("training paused at its time budget - run the notebook again to continue")

    def merge(self):
        if self.exists(f"{self.merged}/config.json"):
            print("merged model already exists")
            return
        if self.hero_answered:
            print("the fine-tuned model's answers are already saved - no merge needed")
            return
        if not self.train_done and not self.dry:
            raise StopPipeline("no trained adapter yet - run the 'train' stage first")
        self.sh([PY, "src/merge_lora.py", "--adapter", f"{self.cfg['train']['out']}/final_adapter",
                 "--out", self.merged])

    def infer_cmd(self, run: str) -> list[str]:
        i = self.cfg["infer"]
        mode = "finetuned" if run == "slm_finetuned" else "fewshot"
        model = self.merged if run == "slm_finetuned" else self.cfg["model"]["base"]
        cmd = [PY, "src/infer_local.py", "--model", model, "--run", run, "--mode", mode, "--backend", i["backend"],
               "--gpu_hourly_usd", str(i["gpu_hourly_usd"]), "--latency_n", str(i.get("latency_n", 100)),
               "--hf_batch_size", str(i.get(f"batch_{mode}", 8)),
               "--hf_token_budget", str(i.get(f"token_budget_{mode}", 16000))]
        if mode == "finetuned" and self.cfg["train"].get("short_prompt"):
            cmd.append("--short_prompt")  # must match how the model was trained
        return cmd

    def run_parallel(self, jobs: dict[str, list[str]], gpus: bool = True, poll_seconds: int = 60) -> dict[str, int]:
        """Run jobs at the same time (with gpus=True, one per GPU and each sees only its own);
        output goes to results/<job>.log and its end is printed when the job finishes."""
        procs = {}
        for n, (name, cmd) in enumerate(jobs.items()):
            print(f"$ [{f'GPU {n}' if gpus else name}] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
            if self.dry:
                continue
            log = open(ROOT / f"results/{name}.log", "w", encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(n)} if gpus else dict(os.environ)
            procs[name] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        codes = {}
        while procs:
            time.sleep(poll_seconds)
            for name, (proc, log) in list(procs.items()):
                if proc.poll() is not None:
                    log.close()
                    codes[name] = proc.returncode
                    tail = (ROOT / f"results/{name}.log").read_text(encoding="utf-8", errors="replace")[-1500:]
                    print(f"\n--- {name} finished (exit {proc.returncode}); end of its log:\n{tail}", flush=True)
                    del procs[name]
            if procs:
                print(f"  still running: {', '.join(procs)}  ({(time.time() - self.t0) / 60:.0f} min elapsed)",
                      flush=True)
        return codes

    def infer(self):
        todo = [r for r in ("slm_finetuned", "slm_base_fewshot") if not self.exists(f"results/preds/{r}.meta.json")]
        if not todo:
            print("local inference already done")
            return
        self.need_gpu()
        if "slm_finetuned" in todo and not self.exists(f"{self.merged}/config.json") and not self.dry:
            raise StopPipeline("no merged model yet - run the 'merge' stage first")
        (ROOT / "results").mkdir(exist_ok=True)
        if max(1, self.n_gpus()) >= 2 and len(todo) == 2:
            print("testing both models at the same time, one per GPU (progress is in results/<run>.log)")
            codes = self.run_parallel({run: self.infer_cmd(run) for run in todo})
        else:
            codes = {run: self.sh(self.infer_cmd(run), check=False) for run in todo}
        self.backup_results("local inference")
        failed = [run for run, code in codes.items() if code]
        if failed:
            raise RuntimeError(f"inference failed for {failed} (see the log above); finished runs are saved")

    def api(self):
        if not (self.train_done or self.hero_answered) and not self.dry:
            print("training not finished - API baselines wait, so they are only paid for once")
            return
        # All contestants run at the same time, each with the session time that is left (minus a few minutes
        # for the report); unfinished questions are kept and continued on the next run.
        contestants = self.contestants()
        if not contestants:
            print("no API contestants with an attached key - nothing to do")
            return
        minutes = self.minutes_left(reserve_minutes=15)
        if minutes < 3 and not self.dry:
            print("no session time left for the API runs - run the notebook again to continue them")
            return
        (ROOT / "results").mkdir(exist_ok=True)
        print(f"{len(contestants)} API models answer the same questions at the same time, up to {minutes:.0f} min "
              "(progress is in results/<name>.log)")
        self.run_parallel({c["name"]: self.api_cmd(c, c.get("limit", 500), c["name"], "results/preds")
                           + ["--max_minutes", f"{minutes:.0f}"] for c in contestants}, gpus=False)
        self.backup_results("API runs")

    def evaluate(self):
        if not list((ROOT / "results/preds").glob("*.meta.json")) and not self.dry:
            print("nothing to evaluate yet")
            return
        self.sh([PY, "src/evaluate.py", "--preds_dir", "results/preds", "--out", "results", "--hero", "slm_finetuned"])
        self.backup_results("report")
        print("\nreport: results/summary.md   charts: results/accuracy.png, results/accuracy_vs_cost.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=None, help=f"comma-separated subset of: {','.join(ALL_STAGES)}")
    ap.add_argument("--config", default=str(ROOT / "kaggle" / "config.toml"))
    ap.add_argument("--dry", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    stages = args.stages.split(",") if args.stages else cfg["pipeline"]["stages"]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}; choose from {ALL_STAGES}")

    runner = Runner(cfg, args.dry)
    print(f"stages: {', '.join(stages)}")
    runner.init_hub()
    runner.restore_results()
    t0 = time.time()
    for stage in stages:
        print(f"\n{'=' * 70}\n>>> {stage}   ({(time.time() - t0) / 60:.0f} min elapsed)\n{'=' * 70}", flush=True)
        try:
            getattr(runner, stage)()
        except StopPipeline as e:
            print(f"\nSTOPPED: {e}")
            return  # clean exit so a Kaggle commit still saves its outputs
        except RuntimeError as e:
            sys.exit(f"\nFAILED in stage '{stage}': {e}")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
