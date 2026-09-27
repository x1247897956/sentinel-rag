"""补齐生成阶段失败的条目，并重算生成指标。

用途：LLM 调用偶发 `Connection reset by peer` 时，不必重跑整轮评测（重跑还会换掉
检索轨迹）。本脚本从 `eval/results/traces_*.json` + `eval/snapshot/answers.jsonl`
里恢复每条的上下文，只重试失败条目，然后重算引用/拒答/要点指标并写回结果文件。

用法：
  uv run python scripts/retry_generation.py --tag v2 [--also-judge]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVAL_DIR, get_settings  # noqa: E402
from src.eval import prompts  # noqa: E402
from src.eval.runner import aggregate_generation, _trace_from_dict  # noqa: E402
from src.retrieval.hybrid import Retriever, citation_hallucination, extract_citations  # noqa: E402
from src.retrieval.models import Embedder, LLMClient, Reranker  # noqa: E402

SNAPSHOT = EVAL_DIR / "snapshot"
RESULTS = EVAL_DIR / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--also-judge", action="store_true")
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()

    answers = load_jsonl(SNAPSHOT / "answers.jsonl")
    judge = {j["qid"]: j for j in (load_jsonl(SNAPSHOT / "judge.jsonl") if (SNAPSHOT / "judge.jsonl").exists() else [])}
    items = {r["qid"]: r for r in load_jsonl(EVAL_DIR / "dataset" / "regression_set.jsonl")}

    conn = None
    retriever = None
    llm = LLMClient(model=get_settings().gen_model)
    settings = get_settings()
    judge_llm = LLMClient(model=settings.judge_model) if settings.judge_model else llm

    failed = [a for a in answers if (a.get("answer") or "").startswith("[生成失败]")]
    print(f"待重试 {len(failed)} 条：{[a['qid'] for a in failed]}")
    if failed and conn is None:
        from src.retrieval import storage

        conn = storage.connect()
        retriever = Retriever(conn, embedder=Embedder(), reranker=Reranker())

    fixed = 0
    for a in failed:
        qid = a["qid"]
        item = items[qid]
        # 重新检索一次拿上下文（检索是确定性的、毫秒级；比依赖轨迹文件更稳）
        trace = retriever.retrieve(item["question"], mode="hybrid_rerank")
        for attempt in range(1, args.attempts + 1):
            try:
                if trace.refused:
                    gen = {
                        "answer": prompts.REFUSAL_TEXT,
                        "citations": [],
                        "refused": True,
                        "allowed_ids": [],
                        "points_partial": None,
                        "points_full": None,
                    }
                else:
                    blocks = [
                        f"[{c.chunk_id}] 来源={c.meta.get('source_type')} 文档={c.meta.get('doc_id')}\n{c.text}"
                        for c in trace.context
                    ]
                    resp = llm.chat(prompts.build_answer_messages(item["question"], blocks),
                                    temperature=0.0, max_tokens=700)
                    answer = (resp["choices"][0]["message"]["content"] or "").strip()
                    citations = extract_citations(answer)
                    gen = {
                        "answer": answer,
                        "citations": citations,
                        "refused": False,
                        "allowed_ids": [c.chunk_id for c in trace.context],
                        "hallucinated": citation_hallucination(citations, trace.retrieved),
                        "prompt_tokens": resp.get("_usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": resp.get("_usage", {}).get("completion_tokens", 0),
                    }
                a.update(
                    {
                        "answer": gen["answer"],
                        "citations": gen["citations"],
                        "allowed_ids": gen["allowed_ids"],
                        "refused": gen["refused"],
                        "gen_model": llm.model,
                        "retried": attempt,
                    }
                )
                fixed += 1
                print(f"  [{qid}] 第 {attempt} 次成功：{gen['answer'][:60]}")
                if args.also_judge and item.get("gold_points") and not gen["refused"]:
                    verdict = judge_llm.json(
                        prompts.build_judge_messages(item["gold_points"], gen["answer"]),
                        temperature=0.0,
                        max_tokens=2500,
                    )
                    covered = [bool(x) for x in verdict.get("covered", [])]
                    n = len(item["gold_points"])
                    covered += [False] * max(0, n - len(covered))
                    judge[qid] = {
                        "qid": qid,
                        "points_partial": round(sum(1 for c in covered[:n] if c) / n, 4),
                        "points_full": 1.0 if all(covered[:n]) else 0.0,
                        "faithful": bool(verdict.get("faithful", True)),
                        "reason": str(verdict.get("reason", ""))[:300],
                        "method": "llm_judge",
                        "judge_model": judge_llm.model,
                    }
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [{qid}] 第 {attempt} 次失败：{str(exc)[:100]}")
                time.sleep(3 * attempt)

    with open(SNAPSHOT / "answers.jsonl", "w", encoding="utf-8") as fh:
        for a in answers:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    if judge:
        with open(SNAPSHOT / "judge.jsonl", "w", encoding="utf-8") as fh:
            for j in sorted(judge.values(), key=lambda x: x["qid"]):
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")

    # 重算生成指标，写回结果文件
    # 复用 runner 的聚合逻辑：这里按 answers 的字段重新组装等价的行
    rows = []
    for a in answers:
        rows.append(
            {
                **a,
                "cite_hit": (1.0 if set(a.get("citations", [])) & set(a.get("gold_chunk_ids", [])) else 0.0)
                if (a.get("citations") and a.get("answerable", True) and not a.get("refused"))
                else None,
                "cite_halluc": bool(
                    [c for c in a.get("citations", []) if c not in set(a.get("allowed_ids", []))]
                ),
                "gen_latency_ms": 0,
                "latency_ms": 0,
            }
        )
        if a["qid"] in judge:
            rows[-1]["points_partial"] = judge[a["qid"]]["points_partial"]
            rows[-1]["points_full"] = judge[a["qid"]]["points_full"]
            rows[-1]["judge_faithful"] = judge[a["qid"]].get("faithful")
    metrics = aggregate_generation(rows)
    out_path = RESULTS / f"eval_{args.tag}.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["results"]["hybrid_rerank"]["generation"] = metrics
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"重试成功 {fixed}/{len(failed)}")
    print("生成指标：" + json.dumps(metrics, ensure_ascii=False))
    if conn is not None:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
