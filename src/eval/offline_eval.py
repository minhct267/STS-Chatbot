from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_groq import ChatGroq

from src.config import LOG_DIR, OLLAMA_MODEL_NAME
from src.logging_utils import INTERACTION_LOG_PATH

EVAL_RESULTS_PATH = LOG_DIR / "eval_results.jsonl"

@dataclass
class EvalResult:
    log_index: int
    session_id: str
    model_name: str
    question: str
    correctness_score: Optional[float]
    grounded: Optional[bool]
    coverage_score: Optional[float]
    coherence_score: Optional[float]
    raw_judgement: str


def iter_interaction_logs(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Interaction log file not found: {path}. "
            "Run the chatbot for a while to generate logs first.",
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_context_from_log(record: Dict[str, Any]) -> str:
    docs = record.get("retrieved_docs") or []
    lines: List[str] = []
    for idx, d in enumerate(docs, start=1):
        source = d.get("source", "unknown")
        page = d.get("page")
        header = f"[{idx}] Source: {source}"
        if page is not None:
            header += f" (page {page})"
        content = d.get("content", "")
        lines.append(f"{header}\n{content}")
    return "\n\n".join(lines)


def build_eval_prompt(record: Dict[str, Any]) -> str:
    question = record.get("question", "")
    answer = record.get("answer", "")
    context = build_context_from_log(record)

    prompt = f"""You are a judge evaluating the quality of answers in a RAG system.

Your task:
- Evaluate the answer based only on the provided CONTEXT (do not use outside knowledge).
- Score according to these criteria:
  1. correctness_score: How correct the answer is compared with the context, in range 0.0–1.0
  2. grounded: Whether the answer is grounded in the context (true/false)
  3. coverage_score: How well the answer covers important information in the context (0.0–1.0)
  4. coherence_score: How coherent, clear and easy to understand the answer is (0.0–1.0)

Respond **only** with JSON in the format:
{{
  "correctness_score": float,
  "grounded": true/false,
  "coverage_score": float,
  "coherence_score": float,
  "comment": "Short explanation in English"
}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""
    return prompt


def parse_eval_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return {}


def evaluate_record(
    idx: int,
    record: Dict[str, Any],
    llm: ChatGroq,
) -> EvalResult:
    prompt = build_eval_prompt(record)
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))
    data = parse_eval_json(text)

    def to_float(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    correctness = to_float(data.get("correctness_score"))
    coverage = to_float(data.get("coverage_score"))
    coherence = to_float(data.get("coherence_score"))
    grounded_raw = data.get("grounded")
    grounded = None
    if isinstance(grounded_raw, bool):
        grounded = grounded_raw
    elif isinstance(grounded_raw, str):
        grounded = grounded_raw.strip().lower() in {"true", "yes", "1"}

    return EvalResult(
        log_index=idx,
        session_id=str(record.get("session_id", "")),
        model_name=str(record.get("model_name", "")),
        question=str(record.get("question", "")),
        correctness_score=correctness,
        grounded=grounded,
        coverage_score=coverage,
        coherence_score=coherence,
        raw_judgement=text,
    )


def write_eval_result(result: EvalResult) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "log_index": result.log_index,
        "session_id": result.session_id,
        "model_name": result.model_name,
        "question": result.question,
        "correctness_score": result.correctness_score,
        "grounded": result.grounded,
        "coverage_score": result.coverage_score,
        "coherence_score": result.coherence_score,
        "raw_judgement": result.raw_judgement,
    }
    with EVAL_RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline self-eval for STS-Chatbot based on interaction logs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of first interactions to evaluate (default: 50).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL_NAME,
        help="Groq model name to use for self-eval (default: same as chat model).",
    )
    args = parser.parse_args()

    print(f"Reading logs from: {INTERACTION_LOG_PATH}")
    logs_iter = iter_interaction_logs(INTERACTION_LOG_PATH)

    llm = ChatGroq(model=args.model, temperature=0.0)

    processed = 0
    scored = 0
    correctness_sum = 0.0

    for idx, record in enumerate(logs_iter):
        if args.limit and processed >= args.limit:
            break
        processed += 1

        try:
            result = evaluate_record(idx, record, llm)
            write_eval_result(result)
            if result.correctness_score is not None:
                correctness_sum += result.correctness_score
                scored += 1
            print(
                f"[{processed}] correctness={result.correctness_score} "
                f"grounded={result.grounded}",
            )
        except Exception as exc:
            print(f"[WARN] Error when evaluating log index {idx}: {exc}")
            continue

    print(f"Processed {processed} interactions, results saved to: {EVAL_RESULTS_PATH}")
    if scored > 0:
        avg = correctness_sum / scored
        print(f"Average correctness over {scored} samples: {avg:.3f}")


if __name__ == "__main__":
    main()
