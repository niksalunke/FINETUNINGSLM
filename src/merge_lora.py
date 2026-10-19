"""Merge a LoRA adapter into the fp16 base model so it can be served (and timed) like any normal model.

    python src/merge_lora.py --adapter outputs/qwen3-4b-tatqa/final_adapter --out outputs/qwen3-4b-tatqa-merged
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="defaults to the model the adapter was trained on")
    args = ap.parse_args()

    base = args.base
    if base is None:
        base = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"merging {args.adapter} into {base} (on CPU, fp16)")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"saved merged model to {args.out}")


if __name__ == "__main__":
    main()
