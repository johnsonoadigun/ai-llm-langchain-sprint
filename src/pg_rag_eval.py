from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.pg_rag_cli import ask, PROJECT_ROOT


EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "eval_questions.json"
OUT_FILE = PROJECT_ROOT / "outputs" / "eval_report.json"


def norm(s: str) -> str:
    return (s or "").strip().lower()


def source_name(src: str) -> str:
    # normalize to just filename for easier checks
    try:
        return Path(src).name
    except Exception:
        return src


def contains_any(answer: str, tokens: List[str]) -> List[str]:
    a = norm(answer)
    hits = []
    for t in tokens:
        tt = norm(t)
        if tt and tt in a:
            hits.append(tt)
    return hits


def sources_hit(citations: List[Dict[str, Any]], expected_sources_any: List[str]) -> List[str]:
    got = {source_name(c.get("source", "")) for c in citations}
    exp = {source_name(s) for s in expected_sources_any}
    return sorted(list(got.intersection(exp)))


def run_eval(k: int = 4) -> Dict[str, Any]:
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Missing eval file: {EVAL_FILE}")

    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    results = []
    passed = 0
    started = time.time()

    for case in cases:
        cid = case.get("id", "unknown")
        q = case["question"]
        expect_any = case.get("expect_any", [])
        expect_sources_any = case.get("expect_sources_any", [])

        res = ask(q, k=k)
        answer = res.get("answer", "")
        citations = res.get("citations", [])

        matched_tokens = contains_any(answer, expect_any)
        matched_sources = sources_hit(citations, expect_sources_any) if expect_sources_any else []

        token_pass = (len(expect_any) == 0) or (len(matched_tokens) > 0)
        source_pass = (len(expect_sources_any) == 0) or (len(matched_sources) > 0)

        case_pass = token_pass and source_pass

        results.append(
            {
                "id": cid,
                "question": q,
                "passed": case_pass,
                "token_pass": token_pass,
                "source_pass": source_pass,
                "matched_tokens": matched_tokens,
                "expected_any": [norm(x) for x in expect_any],
                "matched_sources": matched_sources,
                "expected_sources_any": [source_name(x) for x in expect_sources_any],
                "answer": answer,
                "citations": citations,
            }
        )

        if case_pass:
            passed += 1

    duration_s = round(time.time() - started, 3)
    total = len(results)

    report = {
        "status": "ok",
        "k": k,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 3),
        "duration_seconds": duration_s,
        "results": results,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    report = run_eval(k=4)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
