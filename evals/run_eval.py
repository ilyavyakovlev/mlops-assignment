"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implementation (Phase 5) -----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    history_correct[i] is whether the SQL produced at history step i
    (generate_sql=0, first revise=1, ...) matched the gold result set.
    final_correct mirrors history_correct[-1].
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)
    if not gold_ok:
        return {
            "question": question["question"],
            "db_id": db_id,
            "gold_sql": gold_sql,
            "agent_sql": "",
            "iterations": 0,
            "history_correct": [False],
            "final_correct": False,
            "latency_seconds": 0.0,
            "error": f"gold_sql_error: {gold_err}",
        }

    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "question": question["question"],
            "db_id": db_id,
            "gold_sql": gold_sql,
            "agent_sql": "",
            "iterations": 0,
            "history_correct": [False],
            "final_correct": False,
            "latency_seconds": time.monotonic() - t0,
            "error": f"agent_error: {e}",
        }
    latency = time.monotonic() - t0

    history = data.get("history", [])
    agent_sql = data.get("sql", "")
    iterations = data.get("iterations", 0)

    # Score the SQL produced at each history step independently.
    history_correct: list[bool] = []
    for entry in history:
        _, iter_rows, _ = run_sql(db_id, entry.get("sql", ""))
        history_correct.append(matches(gold_rows, iter_rows))

    # Guard: history should never be empty given a successful response,
    # but fall back to scoring the final sql directly if it is.
    if not history_correct:
        _, final_rows, _ = run_sql(db_id, agent_sql)
        history_correct = [matches(gold_rows, final_rows)]

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "iterations": iterations,
        "history_correct": history_correct,
        "final_correct": history_correct[-1],
        "latency_seconds": latency,
        "error": None,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0, "overall_pass_rate": 0.0, "per_iteration_pass_rate": {}}

    max_iter = max(len(r["history_correct"]) for r in results) - 1

    per_iter: dict[str, float] = {}
    for k in range(max_iter + 1):
        correct = 0
        for r in results:
            # Carry-forward: use last available index if agent stopped early.
            j = min(k, len(r["history_correct"]) - 1)
            if r["history_correct"][j]:
                correct += 1
        per_iter[str(k)] = round(correct / n, 4)

    overall = sum(r["final_correct"] for r in results) / n

    return {
        "n": n,
        "overall_pass_rate": round(overall, 4),
        "per_iteration_pass_rate": per_iter,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
