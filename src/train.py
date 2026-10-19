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
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
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
    def __init__(self, rows, tokenizer, max_len):
        suffix = assistant_suffix(tokenizer)
        self.examples, dropped = [], 0
        for r in rows:
            prompt = tokenizer.apply_chat_template(
                build_messages(r["user_prompt"]), tokenize=False, add_generation_prompt=True)
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
    train_ds = SFTDataset(read_jsonl(Path(args.data) / "train.jsonl", args.max_train_samples), tokenizer, args.max_len)
    val_ds = SFTDataset(read_jsonl(Path(args.data) / "val.jsonl", args.val_samples), tokenizer, args.max_len)

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    on_gpu = torch.cuda.is_available()  # CPU path exists only for smoke tests (needs --no_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if on_gpu else torch.float32, quantization_config=quant,
        device_map={"": local_rank} if on_gpu else "cpu", attn_implementation="sdpa")
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
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

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum * world))
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=on_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,  # batches of similar length -> far less padding
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
        seed=args.seed,
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )
    if is_main:
        print(f"GPUs={world}  train={len(train_ds)}  ~{steps_per_epoch} optimizer steps/epoch  "
              f"effective batch={args.batch_size * args.grad_accum * world}")

    budget = None
    callbacks = []
    if args.max_hours:
        budget = TimeBudget(args.max_hours, torch.device("cuda", local_rank) if on_gpu else torch.device("cpu"))
        callbacks.append(budget)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tokenizer.pad_token_id), callbacks=callbacks)
    resume = resolve_resume(args, targs)
    if is_main:
        print(f"resuming from {resume}" if resume else "starting fresh")
    trainer.train(resume_from_checkpoint=resume)

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
