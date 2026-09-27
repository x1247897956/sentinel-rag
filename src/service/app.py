"""FastAPI 服务：POST /ask（带引用回答）、POST /ingest（增量入库）、GET /health。

三种检索配置都可通过 `mode` 切换，便于在线对照演示：
  vector / fts / hybrid（RRF）/ hybrid_rerank（默认，最终方案）
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    CONTEXT_TOKEN_BUDGET,
    EMBED_MODEL,
    FINAL_TOPN,
    RECALL_TOPK,
    REFUSAL_MIN_COSINE,
    REFUSAL_MIN_FTS,
    RERANK_MODEL,
    get_settings,
)
from src.eval.metrics import percentile
from src.eval.prompts import PROMPT_VERSION
from src.retrieval import storage
from src.retrieval.hybrid import MODES, Retriever
from src.retrieval.models import Embedder, LLMClient, Reranker

app = FastAPI(title="SentinelRAG", description="安全知识库检索 Agent：混合检索 + 引用溯源 + 拒答")

_state: dict[str, Any] = {}


def _retriever() -> Retriever:
    if "retriever" not in _state:
        conn = storage.connect()
        _state["conn"] = conn
        _state["retriever"] = Retriever(conn, embedder=Embedder(), reranker=Reranker())
        _state["llm"] = LLMClient(model=get_settings().gen_model)
    return _state["retriever"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="hybrid_rerank", description="vector | fts | hybrid | hybrid_rerank")
    topk_recall: int = RECALL_TOPK
    topn_final: int = FINAL_TOPN
    token_budget: int = CONTEXT_TOKEN_BUDGET
    refusal_min_fts: float = REFUSAL_MIN_FTS
    refusal_min_cosine: float = REFUSAL_MIN_COSINE
    generate: bool = True


class AskResponse(BaseModel):
    question: str
    mode: str
    refused: bool
    answer: str
    citations: list[str]
    hallucinated_citations: list[str]
    retrieved: list[dict]
    context: list[str]
    truncated: list[str]
    top_score: float
    retrieval_ms: int
    rerank_ms: int
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    prompt_version: str
    models: dict[str, str]


@app.get("/health")
def health() -> dict:
    conn = storage.connect()
    try:
        with conn.cursor() as cur:
            stats = storage.stats(cur)
    finally:
        conn.close()
    settings = get_settings()
    return {
        "status": "ok",
        "chunks": stats["chunks"],
        "documents": stats["documents"],
        "chunks_embedded": stats["chunks_embedded"],
        "documents_by_source": stats["documents_by_source"],
        "models": {
            "embed": EMBED_MODEL,
            "rerank": RERANK_MODEL,
            "generate": settings.gen_model,
        },
        "llm_key_configured": bool(settings.deepseek_api_key),
        "prompt_version": PROMPT_VERSION,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode 必须是 {list(MODES)} 之一")
    retriever = _retriever()
    trace = retriever.retrieve(
        req.question,
        mode=req.mode,
        topk_recall=req.topk_recall,
        topn_final=req.topn_final,
        token_budget=req.token_budget,
        refusal_min_fts=req.refusal_min_fts,
        refusal_min_cosine=req.refusal_min_cosine,
    )
    from src.eval.prompts import REFUSAL_TEXT

    if not req.generate:
        return AskResponse(
            question=req.question,
            mode=req.mode,
            refused=trace.refused,
            answer="" if not trace.refused else REFUSAL_TEXT,
            citations=[],
            hallucinated_citations=[],
            retrieved=[c.to_dict() | {"text": c.text[:400]} for c in trace.candidates[:10]],
            context=[c.chunk_id for c in trace.context],
            truncated=[c.chunk_id for c in trace.candidates if c.truncated],
            top_score=trace.top_score,
            retrieval_ms=trace.total_ms,
            rerank_ms=trace.rerank_ms,
            latency_ms=trace.total_ms,
            prompt_tokens=0,
            completion_tokens=0,
            prompt_version=PROMPT_VERSION,
            models={"embed": EMBED_MODEL, "rerank": RERANK_MODEL},
        )

    llm = _state.get("llm") or LLMClient(model=get_settings().gen_model)
    gen = retriever.answer(req.question, trace, llm)
    from src.retrieval.hybrid import citation_hallucination

    halluc = citation_hallucination(gen["citations"], trace.retrieved)
    latency = trace.total_ms + gen["latency_ms"]
    # 每次问答落 runs 表：模型 + prompt 版本一起落库，否则指标变化无法归因
    conn = _state.get("conn") or storage.connect()
    try:
        with conn.cursor() as cur:
            storage.log_run(
                cur,
                {
                    "run_id": str(uuid.uuid4()),
                    "question": req.question,
                    "config": req.mode,
                    "retrieved": [c.to_dict() for c in trace.candidates[:20]],
                    "answer": gen["answer"],
                    "citations": gen["citations"],
                    "prompt_version": PROMPT_VERSION,
                    "embed_model": EMBED_MODEL,
                    "rerank_model": RERANK_MODEL if req.mode == "hybrid_rerank" else None,
                    "gen_model": llm.model,
                    "latency_ms": latency,
                    "retrieval_ms": trace.total_ms,
                    "rerank_ms": trace.rerank_ms,
                    "prompt_tokens": gen["usage"].get("prompt_tokens", 0),
                    "completion_tokens": gen["usage"].get("completion_tokens", 0),
                    "refused": gen["refused"],
                },
            )
            conn.commit()
    finally:
        pass

    return AskResponse(
        question=req.question,
        mode=req.mode,
        refused=gen["refused"],
        answer=gen["answer"],
        citations=gen["citations"],
        hallucinated_citations=halluc,
        retrieved=[c.to_dict() for c in trace.candidates[:10]],
        context=[c.chunk_id for c in trace.context],
        truncated=[c.chunk_id for c in trace.candidates if c.truncated],
        top_score=trace.top_score,
        retrieval_ms=trace.total_ms,
        rerank_ms=trace.rerank_ms,
        latency_ms=latency,
        prompt_tokens=gen["usage"].get("prompt_tokens", 0),
        completion_tokens=gen["usage"].get("completion_tokens", 0),
        prompt_version=PROMPT_VERSION,
        models={"embed": EMBED_MODEL, "rerank": RERANK_MODEL, "generate": llm.model},
    )


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="服务端 JSONL 路径；不传则读 data/corpus/*.jsonl")
    all_docs: bool = False


@app.post("/ingest")
def ingest(req: IngestRequest | None = None) -> dict:
    """增量入库：内容没变直接跳过；变化则旧 chunk 打 stale。"""
    from src.ingest.pipeline import build_chunks, load_corpus

    req = req or IngestRequest()
    docs = load_corpus()
    docs, chunks = build_chunks(docs)
    from src.retrieval.models import Embedder as _E

    vectors = _E().encode([c.text for c in chunks])
    conn = storage.connect()
    counters = {"inserted_docs": 0, "unchanged_docs": 0, "updated_docs": 0, "inserted_chunks": 0}
    try:
        with conn.cursor() as cur:
            if req.all_docs:
                cur.execute("TRUNCATE runs, chunks, documents RESTART IDENTITY CASCADE")
            idx = 0
            for doc in docs:
                rec = doc.to_record()
                action = storage.upsert_document(cur, rec)
                counters["unchanged_docs" if action == "unchanged" else f"{action}_docs"] += 1
                n_doc_chunks = sum(1 for c in chunks if c.doc_id == doc.doc_id)
                if action != "unchanged":
                    counters["inserted_chunks"] += storage.insert_chunks(
                        cur, rec, chunks[idx : idx + n_doc_chunks], vectors[idx : idx + n_doc_chunks]
                    )
                idx += n_doc_chunks
            conn.commit()
            stats = storage.stats(cur)
    finally:
        conn.close()
    return {"counters": counters, "db": stats}


@app.get("/runs")
def runs(limit: int = 20) -> dict:
    conn = storage.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, question, config, latency_ms, retrieval_ms, refused, prompt_tokens,
                       completion_tokens, created_at
                FROM runs ORDER BY created_at DESC LIMIT %s
                """,
                (limit,),
            )
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.execute("SELECT latency_ms FROM runs WHERE latency_ms IS NOT NULL")
            lat = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    for r in rows:
        if hasattr(r.get("created_at"), "isoformat"):
            r["created_at"] = r["created_at"].isoformat()
    return {
        "runs": rows,
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "n_runs": len(lat),
    }
