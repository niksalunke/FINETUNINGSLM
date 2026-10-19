"""Shared prompt format, target construction, output parsing and safe arithmetic for TAT-QA.

Every model (fine-tuned SLM, base SLM, flagship API LLMs) is held to the same output contract:
a single JSON object. Arithmetic questions return an `equation` that Python executes, so no
model is judged on mental arithmetic -- only on understanding *what* to compute.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

SCALES = ["", "thousand", "million", "billion", "percent"]
ANSWER_TYPES = ["span", "multi-span", "arithmetic", "count"]

SYSTEM_PROMPT = """You are a financial analyst. Answer the question using ONLY the table and paragraphs from the financial report you are given.

Return one JSON object, in exactly one of these forms:
- span (answer copied from the table or text): {"answer_type": "span", "answer": ["<exact text>"], "scale": "<scale>"}
- multi-span (several items copied from the context): {"answer_type": "multi-span", "answer": ["<item 1>", "<item 2>"], "scale": "<scale>"}
- arithmetic (a calculation is needed): {"answer_type": "arithmetic", "equation": "<arithmetic expression>", "scale": "<scale>"}
- count (counting items): {"answer_type": "count", "items": ["<item 1>", "<item 2>"], "scale": ""}

Rules:
- "scale" is one of "", "thousand", "million", "billion", "percent" and is the unit of the answer as given in the report (e.g. a table "in thousands" -> "thousand"; a percentage -> "percent").
- "equation" uses only numbers, + - * / and parentheses. Write plain numbers without commas, currency symbols or units. It is evaluated by Python and must produce the final number in the stated scale (for a percentage change write ((new - old) / old) * 100 with scale "percent").
- Copy span answers exactly as they appear in the context."""

# The fine-tuned SLM learns the full contract from 13k examples, so it only needs a one-line reminder.
# This removes ~300 tokens from every training example (about a third of the compute) and from every
# question it answers later. Prompted models (base SLM, API LLMs) always get the full SYSTEM_PROMPT.
SYSTEM_PROMPT_SHORT = ("Answer from the financial report as one JSON object with answer_type, "
                       "then answer, equation or items, and scale.")

# Extra instruction for models that are prompted (not fine-tuned): they may reason first.
REASONING_SUFFIX = """

You may think step by step first. Then finish with the final JSON object on its own, inside a ```json code block. The last JSON object in your reply is taken as your answer."""


# ----------------------------------------------------------------------------- prompt building

def format_table(table: list[list[str]]) -> str:
    rows = []
    for i, row in enumerate(table):
        cells = [str(c).replace("|", "/").replace("\n", " ").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "---|" * len(cells))
    return "\n".join(rows)


def format_context(item: dict) -> str:
    paragraphs = sorted(item["paragraphs"], key=lambda p: p["order"])
    para_text = "\n".join(f"[{p['order']}] {p['text'].strip()}" for p in paragraphs)
    return f"Table:\n{format_table(item['table']['table'])}\n\nParagraphs:\n{para_text}"


def build_user_prompt(context: str, question: str) -> str:
    return f"{context}\n\nQuestion: {question}"


def build_messages(user_prompt: str, few_shot: list[dict] | None = None, reasoning: bool = False,
                   short: bool = False) -> list[dict]:
    """Chat messages in OpenAI/HF format. `few_shot` items have keys `user` and `assistant`.
    `short` = the one-line instruction used to train and run the fine-tuned model."""
    system = SYSTEM_PROMPT_SHORT if short else SYSTEM_PROMPT + (REASONING_SUFFIX if reasoning else "")
    messages = [{"role": "system", "content": system}]
    for ex in few_shot or []:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


# ----------------------------------------------------------------------------- safe arithmetic

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    """Evaluate a pure arithmetic expression without `eval`. Raises ValueError otherwise."""
    if len(expr) > 500:
        raise ValueError("expression too long")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    try:
        return _eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        raise ValueError(str(e)) from e


def clean_equation(expr: str) -> str:
    """Normalise human-written equations (TAT-QA derivations or model output) into Python arithmetic."""
    s = str(expr)
    s = s.replace("$(", "-(")  # accounting negatives like $(20,345)
    for a, b in {"[": "(", "]": ")", "{": "(", "}": ")", "×": "*", "÷": "/", "−": "-", "–": "-",
                 "$": "", "€": "", "£": "", "¥": ""}.items():
        s = s.replace(a, b)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # thousands separators
    s = re.sub(r"(\d)\s*%", r"\1", s)        # 3.8% -> 3.8 (TAT-QA convention: percent figures used as-is)
    s = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", s)
    return s.strip()


def eval_equation(expr: str) -> float:
    return safe_eval(clean_equation(expr))


# ----------------------------------------------------------------------------- training targets

def _close(a: float, b: float) -> bool:
    return abs(round(a, 2) - round(b, 2)) <= 0.011 or abs(a - b) < 1e-6


def build_target(q: dict) -> dict | None:
    """Gold JSON target for a TAT-QA question, or None if it cannot be expressed faithfully."""
    atype, scale = q["answer_type"], q["scale"]
    if atype in ("span", "multi-span"):
        return {"answer_type": atype, "answer": [str(a) for a in q["answer"]], "scale": scale}
    if atype == "count":
        items = [s.strip() for s in q.get("derivation", "").split("##") if s.strip()]
        if not items or len(items) != int(float(q["answer"])):
            return None
        return {"answer_type": "count", "items": items, "scale": ""}
    if atype == "arithmetic":
        try:
            gold = float(q["answer"])
            eq = clean_equation(q["derivation"])
            value = safe_eval(eq)
        except (ValueError, TypeError):
            return None
        if _close(value, gold):
            return {"answer_type": "arithmetic", "equation": eq, "scale": scale}
        if _close(value * 100, gold):  # ratios that TAT-QA reports as percentages
            return {"answer_type": "arithmetic", "equation": f"({eq}) * 100", "scale": scale}
        return None
    return None


def target_to_text(target: dict) -> str:
    return json.dumps(target, ensure_ascii=False)


# ----------------------------------------------------------------------------- output parsing

def _json_objects(text: str) -> list[str]:
    """All top-level {...} substrings (brace-balanced, string-aware)."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                objs.append(text[start:i + 1])
    return objs


def parse_output(text: str) -> dict | None:
    """Return the last valid JSON object in the model output that looks like an answer."""
    if not text:
        return None
    for candidate in reversed(_json_objects(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("answer" in obj or "equation" in obj or "items" in obj):
            return obj
    return None


def format_number(value: float) -> str:
    """Round half-up to 2 decimals (the official scorer compares at 2 decimals) and return a string.

    A string because the official scorer treats a numeric 0 as "no prediction"; half-up via the
    shortest repr because Python's round(2.485, 2) gives 2.48 on binary floats.
    """
    try:
        return str(Decimal(repr(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError) as e:  # inf, nan, absurdly large
        raise ValueError(f"unformattable number {value!r}") from e


def to_prediction(obj: dict | None) -> tuple[Any, str, str | None]:
    """Turn a parsed model answer into (answer, scale, error) in official TAT-QA prediction format."""
    if obj is None:
        return None, "", "no_answer_json"
    scale = str(obj.get("scale", "") or "").strip().lower()
    if scale not in SCALES:
        scale = next((s for s in SCALES[1:] if s in scale), "")
    atype = obj.get("answer_type", "")

    if "equation" in obj and (atype == "arithmetic" or "answer" not in obj):
        try:
            return format_number(eval_equation(obj["equation"])), scale, None
        except ValueError:
            if "answer" not in obj:
                return None, scale, "bad_equation"
    if "items" in obj and (atype == "count" or "answer" not in obj):
        items = obj["items"] if isinstance(obj["items"], list) else [obj["items"]]
        return str(len(items)), scale, None

    answer = obj.get("answer")
    if answer is None:
        return None, scale, "no_answer"
    if isinstance(answer, list):
        answer = [str(a) for a in answer]
        if atype in ("arithmetic", "count") and len(answer) == 1:
            answer = answer[0]
    elif isinstance(answer, (int, float)) and not isinstance(answer, bool):
        try:
            answer = format_number(answer)
        except ValueError:
            return None, scale, "bad_number"
    else:
        answer = [str(answer)] if atype in ("span", "multi-span") else str(answer)
    return answer, scale, None
