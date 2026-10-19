# SLM vs flagship LLMs on financial report QA

**Hypothesis:** not every task needs a huge LLM. A ~4B model fine-tuned on one domain task can match or beat
flagship LLMs on that task, at a small fraction of the cost and latency.

**Task:** [TAT-QA](https://github.com/NExTplusplus/TAT-QA), with about 16.5k questions over real annual-report excerpts (a table plus
paragraphs). Answering requires finding the right cells or sentences, choosing the arithmetic and getting the unit (thousand,
million, percent) right. That is daily analyst work, all the needed information is in the input, and an official scorer exists.

**Hero model:** `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), QLoRA fine-tuned on Kaggle 2x T4.
**Opponents:** the same base model prompted with worked examples, plus flagship API models (Claude Opus 5.5 by default;
OpenAI, Gemini or any OpenAI-compatible host are optional).

## Design: one output contract for every model

Every model answers with one JSON object:

```json
{"answer_type": "span",       "answer": ["$16.7 million"], "scale": ""}
{"answer_type": "multi-span", "answer": ["2019", "2018"],  "scale": ""}
{"answer_type": "arithmetic", "equation": "((131 - 122) / 122) * 100", "scale": "percent"}
{"answer_type": "count",      "items": ["Europe", "North America"], "scale": ""}
```

Arithmetic is run by a safe Python evaluator for **every** model, so the comparison is about understanding *what* to compute,
not mental arithmetic. Scoring uses the **official TAT-QA scorer** (exact match and F1; the scale must be right).

## Fairness rules

1. Every model gets the same test questions, the same contract, the same calculator and the same scorer.
2. Prompted models (the base SLM and the API LLMs) get their best shot: 4 worked examples (one per answer type, from train),
   permission to reason step by step, plus the model's own built-in thinking. Only the fine-tuned model sees the plain prompt.
3. Test set = official TAT-QA **test** split with released gold answers. The SLM never trains on it; dev is used for validation.
4. Contamination favours the big models: TAT-QA has been public since 2021 and may be in their pre-training data.
5. When API runs use `--limit`, the report's **common view** scores every model on exactly the same questions.
6. Cost is measured, not assumed. API cost = real token usage (thinking included) x list price. Local cost = GPU $/hour x the
   measured wall time for the whole batch on one T4.

**Success criteria (fixed before running):**
- **Strong win:** the fine-tuned 4B model beats the flagship on exact match.
- **Efficiency win:** it lands within about 3 EM points of the flagship at 20x or more lower cost or latency.
- Anything else is reported as a loss.

## Files

| File | Purpose |
|---|---|
| `kaggle_slm_vs_llm.ipynb` | The whole pipeline as a Kaggle notebook (2x T4) |
| `src/prepare_data.py` | Downloads TAT-QA and the official scorer; writes `data/{train,val,test}.jsonl` and `few_shot.json` |
| `src/tatqa_common.py` | Prompt format, gold-target construction, JSON parsing, safe equation execution |
| `src/train.py` | QLoRA (or LoRA) fine-tune; fp16, SDPA, loss on answer tokens only, DDP on both T4s |
| `src/merge_lora.py` | Merges the adapter into the base model for serving and timing |
| `src/infer_local.py` | vLLM (or HF) inference; measures batch throughput and single-request latency |
| `src/infer_api.py` | Claude (Anthropic SDK), OpenAI, Gemini or OpenAI-compatible APIs; resumable, logs tokens, latency and cost |
| `src/evaluate.py` | Official scoring, breakdown by answer type, `results/summary.md`, CSVs and charts |

## Run on Kaggle

1. Put this folder on GitHub (and set `REPO_URL` in the notebook) or upload it as a Kaggle Dataset named `finetuningslm`.
2. Notebook settings: **GPU T4 x2**, Internet on. Add secrets: `ANTHROPIC_API_KEY` (plus any other provider keys) and optionally `HF_TOKEN`.
3. Run the cells in order. Do the dry run first. For the full training, use **Save & Run All (Commit)**.

The same steps as commands:

```bash
python src/prepare_data.py --out data
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 src/train.py --out outputs/qwen3-4b-tatqa
python src/merge_lora.py --adapter outputs/qwen3-4b-tatqa/final_adapter --out outputs/qwen3-4b-tatqa-merged
python src/infer_local.py --model outputs/qwen3-4b-tatqa-merged --run slm_finetuned --mode finetuned
python src/infer_local.py --model Qwen/Qwen3-4B-Instruct-2507 --run slm_base_fewshot --mode fewshot
python src/infer_api.py --provider anthropic --model claude-opus-5-5 --run claude_opus_5_5 --limit 500
python src/evaluate.py --hero slm_finetuned
```

## Budget on Kaggle

| Step | Estimate |
|---|---|
| Training, 2 epochs (~12M tokens/epoch, 1.3% of examples over 2,048 tokens are dropped) | roughly 5-7 h on 2x T4 (the dry run shows the real step time) |
| Merge | ~5 min (CPU) |
| SLM inference on 1,663 test questions with vLLM | minutes |
| Claude Opus 5.5 on 500 questions | roughly $15-30, depending on thinking length |

Kaggle gives about 30 GPU hours per week and 12 h per session. If you are short on time, `--epochs 1` is a reasonable first run.

## Known limits

- T4 has no bf16 support. Training and serving run in fp16. If the loss goes NaN, lower `--lr` to 1e-4 or try `--no_4bit`.
- 13 test questions (0.8%) have gold derivations that do not evaluate to the gold answer. They count against every model equally.
  A perfect-answer check scores 99.1% EM overall (99.9% on the questions that can be scored).
- The API runs deliberately have no server-side fallback model, so every answer comes from the model named in the table.
