"""命令行问答：`uv run python -m src.cli ask "问题" --mode hybrid_rerank`"""

from __future__ import annotations

import argparse
import json

from src.config import (
    CONTEXT_TOKEN_BUDGET,
    FINAL_TOPN,
    RECALL_TOPK,
    REFUSAL_MIN_COSINE,
    REFUSAL_MIN_FTS,
    get_settings,
)
from src.eval.prompts import PROMPT_VERSION
from src.retrieval import storage
from src.retrieval.hybrid import MODES, Retriever, citation_hallucination
from src.retrieval.models import Embedder, LLMClient, Reranker


def main() -> int:
    ap = argparse.ArgumentParser(description="SentinelRAG CLI")
    ap.add_argument("command", choices=["ask", "search", "stats", "probe"])
    ap.add_argument("question", nargs="?", default=None)
    ap.add_argument("--mode", default="hybrid_rerank", choices=list(MODES))
    ap.add_argument("--topk-recall", type=int, default=RECALL_TOPK)
    ap.add_argument("--topn-final", type=int, default=FINAL_TOPN)
    ap.add_argument("--token-budget", type=int, default=CONTEXT_TOKEN_BUDGET)
    ap.add_argument("--refusal-min-fts", type=float, default=REFUSAL_MIN_FTS)
    ap.add_argument("--refusal-min-cosine", type=float, default=REFUSAL_MIN_COSINE)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = storage.connect()
    try:
        if args.command == "stats":
            with conn.cursor() as cur:
                print(json.dumps(storage.stats(cur), ensure_ascii=False, indent=2))
            return 0
        if args.command == "probe":
            if not args.question:
                print("probe 需要给一个中文查询")
                return 2
            with conn.cursor() as cur:
                print(json.dumps(storage.fts_probe(cur, args.question), ensure_ascii=False, indent=2))
            return 0

        retriever = Retriever(conn, embedder=Embedder(), reranker=Reranker())
        trace = retriever.retrieve(
            args.question or "",
            mode=args.mode,
            topk_recall=args.topk_recall,
            topn_final=args.topn_final,
            token_budget=args.token_budget,
            refusal_min_fts=args.refusal_min_fts,
            refusal_min_cosine=args.refusal_min_cosine,
        )
        out: dict = {
            "question": args.question,
            "mode": args.mode,
            "top_score": round(trace.top_score, 4),
            "refused": trace.refused,
            "retrieval_ms": trace.total_ms,
            "rerank_ms": trace.rerank_ms,
            "candidates": [c.to_dict() for c in trace.candidates[:10]],
            "context": [c.chunk_id for c in trace.context],
        }
        if args.command == "ask" and not trace.refused:
            llm = LLMClient(model=get_settings().gen_model)
            gen = retriever.answer(args.question or "", trace, llm)
            out["answer"] = gen["answer"]
            out["citations"] = gen["citations"]
            out["hallucinated_citations"] = citation_hallucination(gen["citations"], trace.retrieved)
            out["prompt_version"] = PROMPT_VERSION
        elif trace.refused:
            out["answer"] = "知识库中没有依据，不能回答该问题。"
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"问题：{out['question']}（配置 {out['mode']}，top_score={out['top_score']}，{out['retrieval_ms']}ms）")
            print(f"回答：{out.get('answer', '(仅检索，未生成)')}")
            if out.get("citations"):
                print(f"引用：{out['citations']}")
            if out.get("hallucinated_citations"):
                print(f"⚠️ 引用幻觉：{out['hallucinated_citations']}")
            print("送入上下文的 chunk：")
            for c in trace.context:
                print(f"  [{c.chunk_id}] {c.meta.get('section')} rerank={c.rerank_score:.3f}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
