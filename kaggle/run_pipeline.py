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
ALL_STAGES = ["setup", "data", "llm_smoketest", "dryrun", "train", "merge", "infer", "api", "evaluate"]
TRAIN_LIBS = ['transformers>=4.51', 'peft>=0.13', 'accelerate>=1.0', 'bitsandbytes>=0.43', "anthropic", "openai"]


class StopPipeline(Exception):
    """Raised to end the run cleanly (e.g. training paused at its time budget)."""


class Runner:
    def __init__(self, cfg: dict, dry: bool):
        self.cfg = cfg
        self.dry = dry

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
    def merged(self) -> str:
        return f"{self.cfg['train']['out']}-merged"

    def api_cmd(self, provider: str, sec: dict, limit: int, run: str, preds_dir: str) -> list[str]:
        cmd = [PY, "src/infer_api.py", "--provider", provider, "--model", sec["model"],
               "--run", run, "--limit", str(limit), "--preds_dir", preds_dir]
        if provider == "openai_compatible":
            cmd += ["--base_url", sec["base_url"], "--api_key_env", "LLM_API_KEY"]
        if "price_in" in sec and "price_out" in sec:
            cmd += ["--price_in", str(sec["price_in"]), "--price_out", str(sec["price_out"])]
        return cmd + shlex.split(sec.get("extra", ""))

    def llm_ready(self) -> bool:
        sec = self.cfg.get("llm", {})
        if not sec.get("enabled"):
            print("LLM baseline disabled in config")
            return False
        if not os.environ.get("LLM_API_KEY"):
            print("LLM_API_KEY secret not attached - skipping your LLM")
            return False
        return True

    def claude_ready(self) -> bool:
        if not self.cfg.get("claude", {}).get("enabled"):
            return False
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY secret not attached - skipping the Claude baseline")
            return False
        return True

    def train_args(self, out: str) -> list[str]:
        t = self.cfg["train"]
        return ["src/train.py", "--model", self.cfg["model"]["base"], "--data", "data", "--out", out,
                "--lr", str(t["lr"]), "--batch_size", str(t["batch_size"]), "--grad_accum", str(t["grad_accum"]),
                "--max_len", str(t["max_len"])]

    # ------------------------------------------------------------------ stages
    def setup(self):
        self.sh([PY, "-m", "pip", "install", "-q", "-U", *TRAIN_LIBS])
        self.sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
        print(f"GPUs visible: {self.n_gpus()}")

    def data(self):
        if self.exists("data/test.jsonl") and self.exists("data/raw/tatqa_metric.py"):
            print("data already prepared")
            return
        self.sh([PY, "src/prepare_data.py", "--out", "data"])

    def llm_smoketest(self):
        if not self.llm_ready():
            return
        sec = self.cfg["llm"]
        # separate folder so these 5 answers never mix into the real comparison
        self.sh(self.api_cmd("openai_compatible", sec, 5, f"{sec['run']}_smoketest", "results/smoketest"), check=False)

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

    def train(self):
        if self.train_done:
            print("training already finished")
            return
        self.need_gpu()
        t = self.cfg["train"]
        args = self.train_args(t["out"]) + ["--epochs", str(t["epochs"]), "--save_steps", str(t["save_steps"]),
                                             "--max_hours", str(t["max_hours"]), "--resume"]
        if t.get("hub_repo"):
            if os.environ.get("HF_TOKEN"):
                args += ["--hub_repo", t["hub_repo"]]
            else:
                print("hub_repo is set but the HF_TOKEN secret is not attached - checkpoints stay local only")
        self.launch(args)
        if not self.train_done and not self.dry:
            raise StopPipeline("training paused at its time budget - run the notebook again to continue")

    def merge(self):
        if self.exists(f"{self.merged}/config.json"):
            print("merged model already exists")
            return
        if not self.train_done and not self.dry:
            raise StopPipeline("no trained adapter yet - run the 'train' stage first")
        self.sh([PY, "src/merge_lora.py", "--adapter", f"{self.cfg['train']['out']}/final_adapter",
                 "--out", self.merged])

    def _infer(self, model: str, run: str, mode: str):
        if self.exists(f"results/preds/{run}.meta.json"):
            print(f"{run}: already done")
            return
        i = self.cfg["infer"]
        base = [PY, "src/infer_local.py", "--model", model, "--run", run, "--mode", mode,
                "--gpu_hourly_usd", str(i["gpu_hourly_usd"])]
        if i["backend"] == "vllm" and self.sh(base + ["--backend", "vllm"], check=False) == 0:
            return
        if i["backend"] == "vllm":
            print("vLLM run failed - falling back to the plain Hugging Face backend")
        self.sh(base + ["--backend", "hf"])

    def infer(self):
        self.need_gpu()
        if not self.exists(f"{self.merged}/config.json") and not self.dry:
            raise StopPipeline("no merged model yet - run the 'merge' stage first")
        todo = [r for r in ("slm_finetuned", "slm_base_fewshot") if not self.exists(f"results/preds/{r}.meta.json")]
        if not todo:
            print("local inference already done")
            return
        if self.cfg["infer"]["backend"] == "vllm":
            # installed only now: vLLM may swap torch versions, which must not disturb training
            if self.sh([PY, "-m", "pip", "install", "-q", "vllm"], check=False):
                self.cfg["infer"]["backend"] = "hf"
        self._infer(self.merged, "slm_finetuned", "finetuned")
        self._infer(self.cfg["model"]["base"], "slm_base_fewshot", "fewshot")

    def api(self):
        if not self.train_done and not self.dry:
            print("training not finished - API baselines wait, so they are only paid for once")
            return
        if self.llm_ready():
            sec = self.cfg["llm"]
            self.sh(self.api_cmd("openai_compatible", sec, sec["limit"], sec["run"], "results/preds"), check=False)
        if self.claude_ready():
            sec = self.cfg["claude"]
            self.sh(self.api_cmd("anthropic", sec, sec["limit"], sec["run"], "results/preds"), check=False)

    def evaluate(self):
        if not list((ROOT / "results/preds").glob("*.meta.json")) and not self.dry:
            print("nothing to evaluate yet")
            return
        self.sh([PY, "src/evaluate.py", "--preds_dir", "results/preds", "--out", "results", "--hero", "slm_finetuned"])
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
