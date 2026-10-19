"""LoRA / QLoRA fine-tuning of a 3-4B instruct model on TAT-QA, sized for Kaggle 2x T4.

Turing GPUs (T4) have no bf16 and no FlashAttention-2, so everything runs in fp16 with SDPA attention.
Loss is computed on the assistant answer only (prompt tokens are masked).

Single GPU:
    python src/train.py --data data --out outputs/qwen3-4b-tatqa
Both T4s (DDP, ~2x faster):
    accelerate launch --multi_gpu --num_processes 2 src/train.py --data data --out outputs/qwen3-4b-tatqa

Surviving Kaggle's 12 h session limit: pass --max_hours 11 --resume --hub_repo <user>/<repo>.
When the time budget runs out, the script saves a checkpoint (and uploads it to the Hub) and exits cleanly.
Re-running the exact same command continues from the newest checkpoint: the local one if the disk still has it,
otherwise the one on the Hub. Keep the GPU count and batch settings the same when resuming.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on 15 GB T4s

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tatqa_common import build_messages  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/qwen3-4b-tatqa")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=2048, help="examples longer than this are dropped")
    ap.add_argument("--batch_size", type=int, default=2, help="per GPU")
    ap.add_argument("--grad_accum", type=int, default=8, help="effective batch = batch_size * grad_accum * n_gpus")
    ap.add_argument("--no_4bit", action="store_true", help="plain LoRA on fp16 weights instead of QLoRA")
    ap.add_argument("--short_prompt", action="store_true",
                    help="train with the one-line instruction (~300 fewer tokens per example); "
                         "inference must then also use --short_prompt")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="keep activations instead of recomputing them: ~25%% less compute, much more memory")
    ap.add_argument("--max_train_samples", type=int, default=None, help="for quick dry runs")
    ap.add_argument("--val_samples", type=int, default=300)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in --out, else from the Hub (--hub_repo); "
                         "starts fresh if there is none")
    ap.add_argument("--resume_from", default=None, help="resume from this checkpoint directory")
    ap.add_argument("--hub_repo", default=None,
                    help="e.g. username/qwen3-4b-tatqa: push every checkpoint to a private HF repo "
                         "(survives Kaggle session limits; needs HF_TOKEN)")
    ap.add_argument("--max_hours", type=float, default=None,
                    help="stop cleanly (checkpoint saved) after this many hours of training, e.g. 11 on Kaggle")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def assistant_suffix(tokenizer) -> str:
    """Text the chat template appends after an assistant message (e.g. '<|im_end|>\\n' for Qwen)."""
    marker = "XYZZY_ANSWER"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": marker}], tokenize=False)
    return text[text.rindex(marker) + len(marker):]


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len, short_prompt=False):
        suffix = assistant_suffix(tokenizer)
        self.examples, dropped = [], 0
        for r in rows:
            prompt = tokenizer.apply_chat_template(
                build_messages(r["user_prompt"], short=short_prompt), tokenize=False, add_generation_prompt=True)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(r["target"] + suffix, add_special_tokens=False)["input_ids"]
            ids = p_ids + c_ids
            if len(ids) > max_len:
                dropped += 1
                continue
            self.examples.append({"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids})
        lengths = sorted(len(e["input_ids"]) for e in self.examples)
        if lengths:
            print(f"  {len(self.examples)} examples, dropped {dropped} > {max_len} tokens; "
                  f"length p50={lengths[len(lengths) // 2]} p95={lengths[int(len(lengths) * .95)]} max={lengths[-1]}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class TimeBudget(TrainerCallback):
    """Save a checkpoint and stop when the wall-clock budget is used up.

    All DDP ranks must stop on the same step or they deadlock, so the decision is all-reduced."""

    def __init__(self, hours: float, device):
        self.deadline = time.time() + hours * 3600
        self.device = device
        self.hit = False

    def on_step_end(self, args, state, control, **kwargs):
        flag = torch.tensor([1.0 if time.time() >= self.deadline else 0.0], device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            self.hit = True
            control.should_save = True
            control.should_training_stop = True
        return control


def resolve_resume(args, targs) -> str | None:
    """Checkpoint to continue from: --resume_from, else newest local checkpoint, else the Hub's last-checkpoint."""
    if args.resume_from:
        return args.resume_from
    if not args.resume:
        return None
    local = get_last_checkpoint(args.out) if Path(args.out).is_dir() else None
    if local:
        return local
    if args.hub_repo:
        hub_ckpt = Path(args.out) / "last-checkpoint"
        with targs.main_process_first(local=True, desc="download checkpoint from the Hub"):
            if not (hub_ckpt / "trainer_state.json").exists():
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(args.hub_repo, allow_patterns=["last-checkpoint/*"], local_dir=args.out)
                except Exception as e:  # no repo yet / no checkpoint yet -> fresh start
                    print(f"no Hub checkpoint to resume from ({type(e).__name__})")
        if (hub_ckpt / "trainer_state.json").exists():
            return str(hub_ckpt)
    return None


class AnswerOnlyLossTrainer(Trainer):
    """Scores only the positions whose next token is part of the answer.

    The default loss turns every position into a 152k-vocabulary logit row (several GB for a 2 x 1,300-token
    batch), although only the ~30-80 answer tokens are trained. Passing those positions as `logits_to_keep`
    gives the same loss for a fraction of the memory. The loss is a per-batch mean, and the Trainer is told so,
    which keeps gradient-accumulation scaling identical on transformers 4.x and 5.x.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False  # 4.x: divide by grad-accum steps
        self.loss_is_scaled_for_ga = False      # 5.x: same

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs["labels"][:, 1:]                             # position t predicts token t+1
        keep = (targets != -100).any(dim=0).nonzero(as_tuple=True)[0]  # positions any row trains on
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=keep)
        logits = outputs.logits.float()
        kept = targets[:, keep]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), kept.reshape(-1), ignore_index=-100)
        return (loss, outputs) if return_outputs else loss


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("target"):
                rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print("building datasets")
    train_ds = SFTDataset(read_jsonl(Path(args.data) / "train.jsonl", args.max_train_samples), tokenizer, args.max_len,
                          args.short_prompt)
    val_ds = SFTDataset(read_jsonl(Path(args.data) / "val.jsonl", args.val_samples), tokenizer, args.max_len,
                        args.short_prompt)

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    on_gpu = torch.cuda.is_available()  # CPU path exists only for smoke tests (needs --no_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if on_gpu else torch.float32, quantization_config=quant,
        device_map={"": local_rank} if on_gpu else "cpu", attn_implementation="sdpa")
    model.config.use_cache = False
    mem_load_gb = torch.cuda.memory_allocated(local_rank) / 2**30 if on_gpu else 0.0
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    # fp16 AMP needs fp32 master weights for everything that trains.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if is_main:
        model.print_trainable_parameters()

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum * world)))
    # Options that differ between transformers 4.x and 5.x, so the script runs on either:
    #  - warmup as an integer step count (5.x removed warmup_ratio)
    #  - length-grouped batching (5.x replaced group_by_length with train_sampling_strategy)
    ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    compat = {"warmup_steps": max(1, round(0.03 * steps_per_epoch * args.epochs))}
    if "train_sampling_strategy" in ta_fields:
        compat["train_sampling_strategy"] = "group_by_length"
    else:
        compat["group_by_length"] = True
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=on_gpu,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        prediction_loss_only=True,  # evaluation needs the loss only, never the full logits
        seed=args.seed,
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        hub_strategy="checkpoint",
        hub_private_repo=True,
        **compat,  # incl. batches of similar length -> far less padding
    )
    if is_main:
        print(f"GPUs={world}  train={len(train_ds)}  ~{steps_per_epoch} optimizer steps/epoch  "
              f"effective batch={args.batch_size * args.grad_accum * world}")

    budget = None
    callbacks = []
    if args.max_hours:
        budget = TimeBudget(args.max_hours, torch.device("cuda", local_rank) if on_gpu else torch.device("cpu"))
        callbacks.append(budget)
    trainer = AnswerOnlyLossTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tokenizer.pad_token_id), callbacks=callbacks)
    resume = resolve_resume(args, targs)
    if is_main:
        print(f"resuming from {resume}" if resume else "starting fresh")
    result = trainer.train(resume_from_checkpoint=resume)
    if trainer.is_world_process_zero():
        # speed numbers for the pipeline's training-time estimate
        metrics = dict(result.metrics)
        if on_gpu:
            metrics.update(
                gpu_mem_after_load_gb=round(mem_load_gb, 2),
                gpu_mem_peak_gb=round(torch.cuda.max_memory_allocated(local_rank) / 2**30, 2),
                gpu_mem_total_gb=round(torch.cuda.get_device_properties(local_rank).total_memory / 2**30, 2),
                longest_example_tokens=max(len(e["input_ids"]) for e in train_ds.examples),
                max_len=args.max_len, quantize_4bit=not args.no_4bit, batch_size=args.batch_size,
                gradient_checkpointing=not args.no_grad_ckpt, short_prompt=args.short_prompt)
        (Path(args.out) / "train_metrics.json").write_text(json.dumps(metrics, indent=1))

    if budget is not None and budget.hit:
        if trainer.is_world_process_zero():
            last = get_last_checkpoint(args.out)
            print(f"time budget of {args.max_hours} h reached at step {trainer.state.global_step}/"
                  f"{trainer.state.max_steps}; checkpoint: {last}")
            if args.hub_repo and last:
                # Synchronous upload so the checkpoint is safe before the session ends.
                from huggingface_hub import upload_folder
                upload_folder(repo_id=args.hub_repo, folder_path=last, path_in_repo="last-checkpoint",
                              commit_message=f"checkpoint at step {trainer.state.global_step} (time budget)")
                print(f"uploaded to {args.hub_repo}/last-checkpoint")
            print("re-run the same command with --resume to continue")
        return

    if trainer.is_world_process_zero():
        final = Path(args.out) / "final_adapter"
        trainer.model.save_pretrained(final)
        tokenizer.save_pretrained(final)
        (final / "train_config.json").write_text(json.dumps(vars(args), indent=1))
        print(f"saved adapter to {final}")
        if args.hub_repo:
            trainer.model.push_to_hub(args.hub_repo, private=True)
            tokenizer.push_to_hub(args.hub_repo, private=True)


if __name__ == "__main__":
    main()
