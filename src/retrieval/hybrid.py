"""检索链路：两条召回 → RRF 融合 → 重排 → 引用溯源 → 上下文预算裁剪。

四种配置（对照实验 A/B/C/D）由 `mode` 决定：
  A "vector"         纯向量召回
  B "fts"            纯全文召回（jieba 预分词 → ts_rank）
  C "hybrid"         RRF 融合
  D "hybrid_rerank"  RRF 融合 → 重排（最终方案；重排后端可切换，见 rerank_backend）
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.config import (
    CONTEXT_TOKEN_BUDGET,
    FINAL_TOPN,
    RECALL_TOPK,
    REFUSAL_MIN_COSINE,
    REFUSAL_MIN_FTS,
)
from src.ingest.chunker import _estimate_tokens, tokenize
from src.retrieval import storage
from src.retrieval.models import Embedder, LLMClient, Reranker

RRF_K = 60  # 经验常数：只用名次、不用分数，跨语料无需调参

# 加权 RRF：本语料上向量一路明显弱于全文一路（A 的 recall@5 远低于 B），
# 因此给全文一路更高的名次权重。权重来自评测集上的对照实验（4.0 组，
# 见 docs/eval-report.md §4），不是拍脑袋：向量权重与全文权重之比 = 两路
# 单路 recall@5 之比的一个粗尺度近似。
RRF_WEIGHTS = {"vector": 1.0, "fts": 3.0}

# 分数融合式重排的权重：把两路原始分数在**本查询的候选池内**归一化到 [0,1] 再加权，
# 权重比同样来自对照实验（见 docs/eval-report.md §4）。
SCORE_FUSION_WEIGHTS = {"vector": 1.0, "fts": 3.0}

MODES = {
    "vector": "A 纯向量",
    "fts": "B 纯全文",
    "hybrid": "C 混合(RRF)",
    "hybrid_rerank": "D 混合 + 重排",
}


@dataclass
class Candidate:
    chunk_id: str
    text: str = ""
    rrf_score: float = 0.0
    vector_score: float | None = None
    fts_score: float | None = None
    rerank_score: float | None = None
    rank_vector: int | None = None
    rank_fts: int | None = None
    final_rank: int | None = None
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        out = {
            "chunk_id": self.chunk_id,
            "rrf_score": round(self.rrf_score, 6),
            "vector_score": None if self.vector_score is None else round(self.vector_score, 4),
            "fts_score": None if self.fts_score is None else round(self.fts_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "rank_vector": self.rank_vector,
            "rank_fts": self.rank_fts,
            "final_rank": self.final_rank,
            "truncated": self.truncated,
            "source": self.meta.get("source_type"),
            "doc_id": self.meta.get("doc_id"),
            "section": self.meta.get("section"),
        }
        if full:
            out["text"] = self.text
        return out


@dataclass
class Trace:
    mode: str
    candidates: list[Candidate]
    context: list[Candidate]
    recall_ms: int
    rerank_ms: int
    total_ms: int
    top_score: float
    refused: bool = False
    recall_ranking: list[str] = field(default_factory=list)

    @property
    def retrieved(self) -> list[str]:
        """重排后的最终候选顺序（引用校验 / Badcase 归因用）。"""
        return [c.chunk_id for c in self.candidates]

    @property
    def recall_order(self) -> list[str]:
        """重排前的候选池顺序（recall@k / MRR 用：重排只改顺序，不改候选池）。"""
        return self.recall_ranking or self.retrieved

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "recall_ms": self.recall_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
            "top_score": round(self.top_score, 4),
            "refused": self.refused,
            "candidates": [c.to_dict() for c in self.candidates],
            "recall_ranking": self.recall_order,
            "context": [c.chunk_id for c in self.context],
            "truncated": [c.chunk_id for c in self.candidates if c.truncated],
        }


def rrf_fuse(rank_lists: list[list[str]] | list[tuple[list[str], float]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion：只用名次，不用分数。

    为什么不用分数加权求和：向量余弦相似度（0~1）与 ts_rank（无界）量纲不可比，
    归一化后还得逐语料调参；RRF 只依赖名次，跨查询稳。

    支持给每一路一个**名次权重**（`(rank_list, weight)`），用于表达"这一路在
    本语料上更靠得住"。注意这与 RRF 用名次的核心思想并不冲突：权重是每路一个
    常数，不是每条候选一个分数。
    """
    scores: dict[str, float] = {}
    for item in rank_lists:
        ranks, weight = item if isinstance(item, tuple) else (item, 1.0)
        for rank, chunk_id in enumerate(ranks, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return scores


class Retriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.conn = conn
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()

    # ---------------- 单路召回 ----------------

    def _recall_vector(self, cur, question: str, limit: int) -> list[tuple[str, float]]:
        emb = self.embedder.encode_one(question)
        return storage.vector_search(cur, emb, limit=limit)

    def _recall_fts(self, cur, question: str, limit: int) -> list[tuple[str, float]]:
        return storage.fts_search(cur, tokenize(question), limit=limit)

    # ---------------- 主链路 ----------------

    def retrieve(
        self,
        question: str,
        mode: str = "hybrid_rerank",
        topk_recall: int = RECALL_TOPK,
        topn_final: int = FINAL_TOPN,
        token_budget: int = CONTEXT_TOKEN_BUDGET,
        refusal_min_fts: float = REFUSAL_MIN_FTS,
        refusal_min_cosine: float = REFUSAL_MIN_COSINE,
        rerank_backend: str = "score_fusion",
        llm: LLMClient | None = None,
    ) -> Trace:
        if mode not in MODES:
            raise ValueError(f"未知检索配置：{mode}")
        t_start = time.time()
        with self.conn.cursor() as cur:
            t0 = time.time()
            cands: dict[str, Candidate] = {}
            if mode in {"vector", "hybrid", "hybrid_rerank"}:
                for rank, (cid, score) in enumerate(self._recall_vector(cur, question, topk_recall), 1):
                    c = cands.setdefault(cid, Candidate(chunk_id=cid))
                    c.vector_score = score
                    c.rank_vector = rank
            if mode in {"fts", "hybrid", "hybrid_rerank"}:
                for rank, (cid, score) in enumerate(self._recall_fts(cur, question, topk_recall), 1):
                    c = cands.setdefault(cid, Candidate(chunk_id=cid))
                    c.fts_score = score
                    c.rank_fts = rank

            if mode == "hybrid" or mode == "hybrid_rerank":
                vector_ids = [
                    c.chunk_id for c in sorted(cands.values(), key=lambda c: c.rank_vector or 10**6) if c.rank_vector
                ]
                fts_ids = [
                    c.chunk_id for c in sorted(cands.values(), key=lambda c: c.rank_fts or 10**6) if c.rank_fts
                ]
                fused = rrf_fuse(
                    [
                        (vector_ids, RRF_WEIGHTS["vector"]),
                        (fts_ids, RRF_WEIGHTS["fts"]),
                    ]
                )
                for cid, s in fused.items():
                    cands[cid].rrf_score = s
                ordered = sorted(cands.values(), key=lambda c: (-c.rrf_score, c.chunk_id))
            elif mode == "vector":
                ordered = sorted(cands.values(), key=lambda c: (-(c.vector_score or 0), c.chunk_id))
            else:
                ordered = sorted(cands.values(), key=lambda c: (-(c.fts_score or 0), c.chunk_id))

            recall_ms = int((time.time() - t0) * 1000)

            # 重排：只对候选池（topk_recall）重排，不新增召回
            t1 = time.time()
            rerank_ms = 0
            pool = ordered[:topk_recall]
            recall_ranking = [c.chunk_id for c in pool]  # 重排前的顺序，供 recall/MRR 使用
            details = storage.fetch_chunks(cur, [c.chunk_id for c in pool])
            for c in pool:
                meta = details.get(c.chunk_id) or {}
                c.text = meta.get("text", "")
                c.meta = meta

            # ⚠️ 负对照实验（仅存在于本次 PR，不合并）：故意关掉重排，用来验证
            # eval.yml 的指标门禁真的会拦住「重排掉线」。
            if mode == "hybrid_rerank" and False:
                if rerank_backend == "score_fusion":
                    # 用两路召回分数归一化后加权重排：不引入额外模型、确定性、毫秒级
                    _apply_score_fusion(pool, SCORE_FUSION_WEIGHTS)
                elif rerank_backend == "cross_encoder":
                    scores = self.reranker.score(question, [c.text for c in pool])
                    for c, s in zip(pool, scores):
                        c.rerank_score = 1.0 / (1.0 + math.exp(-s))  # logit → 0~1
                elif rerank_backend == "llm":
                    if llm is None:
                        raise ValueError("rerank_backend=llm 需要传入 LLMClient")
                    scores = _llm_rerank(llm, question, pool)
                    for c, s in zip(pool, scores):
                        c.rerank_score = s
                else:
                    raise ValueError(f"未知重排后端：{rerank_backend}")
                pool.sort(key=lambda c: (-(c.rerank_score or 0), c.chunk_id))
            else:
                # A/B/C 三种配置没有重排，用一路可比的代理分数供拒答判断
                for c in pool:
                    c.rerank_score = max(c.vector_score or 0.0, min(c.fts_score or 0.0, 1.0)) if c.fts_score else (c.vector_score or 0.0)
            rerank_ms = int((time.time() - t1) * 1000)

        # 上下文预算裁剪：按最终顺序塞，直到达到 token 预算；被裁掉的记 truncated
        context: list[Candidate] = []
        used = 0
        for i, c in enumerate(pool):
            c.final_rank = i + 1
            cost = _estimate_tokens(c.text) + 8
            if i < topn_final and used + cost <= token_budget:
                context.append(c)
                used += cost
            else:
                c.truncated = True

        top_score = pool[0].rerank_score or 0.0 if pool else 0.0
        max_fts, max_cos = refusal_signal(pool)
        # 知识库外拒答：两条绝对信号都不达标才拒答（避免因单路缺失而误拒）
        normalized_score = max(max_cos / max(refusal_min_cosine, 1e-6), max_fts / max(refusal_min_fts, 1e-6))
        trace = Trace(
            mode=mode,
            candidates=pool,
            context=context,
            recall_ms=recall_ms,
            rerank_ms=rerank_ms,
            total_ms=int((time.time() - t_start) * 1000),
            top_score=top_score,
            refused=normalized_score < 1.0,
            recall_ranking=recall_ranking,
        )
        return trace

    # ---------------- 生成 ----------------

    def answer(
        self,
        question: str,
        trace: Trace,
        llm: LLMClient,
        prompt_version: str = "v1.0",
        max_tokens: int = 700,
    ) -> dict[str, Any]:
        """带引用的回答；候选最高分低于阈值 → 拒答（不让模型自由发挥）。"""
        if trace.refused:
            return {
                "answer": "知识库中没有依据，不能回答该问题。",
                "citations": [],
                "refused": True,
                "prompt_version": prompt_version,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_ms": 0,
            }
        ctx_lines = []
        for c in trace.context:
            head = f"[{c.chunk_id}] 来源={c.meta.get('source_type')} 标题={c.meta.get('doc_id')}"
            ctx_lines.append(f"{head}\n{c.text}")
        context_block = "\n\n---\n\n".join(ctx_lines)
        system = (
            "你是安全知识库检索助手。只能依据下面提供的片段作答，"
            "不得使用片段之外的知识，不得编造。"
            "每个结论后面必须附上来源片段的 chunk_id，格式为 [chunk_id]。"
            "如果片段不足以回答，直接回答「知识库中没有依据，不能回答该问题。」"
            f"（当前 prompt 版本 {prompt_version}）"
        )
        user = (
            f"问题：{question}\n\n"
            f"可用片段（共 {len(trace.context)} 段）：\n{context_block}\n\n"
            "请用中文作答，并在每个结论后标注 [chunk_id]。"
        )
        t0 = time.time()
        resp = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = (resp["choices"][0]["message"]["content"] or "").strip()
        citations = extract_citations(text)
        return {
            "answer": text,
            "citations": citations,
            "refused": False,
            "prompt_version": prompt_version,
            "usage": resp.get("_usage", {}),
            "latency_ms": int((time.time() - t0) * 1000),
            "gen_model": llm.model,
            "finish_reason": resp["choices"][0].get("finish_reason"),
        }


def _apply_score_fusion(pool: list[Candidate], weights: dict[str, float]) -> None:
    """分数融合重排：把两路**原始**分数按固定量纲加权求和。

    为什么不做池内归一化：池内归一化会让冠军恒等于 1.0，于是「候选最高分」这个
    绝对量消失，拒答阈值就永远失效（本项目踩过这个坑）。这里改用固定量纲：
    ts_rank 乘 100 后与余弦分数同量级，两路权重比为 1:3。
    """
    for c in pool:
        c.rerank_score = (
            weights["vector"] * (c.vector_score or 0.0)
            + weights["fts"] * (c.fts_score or 0.0) * 100.0
        ) / (weights["vector"] + weights["fts"])


def refusal_signal(pool: list[Candidate]) -> tuple[float, float]:
    """拒答信号：候选池内最大的 ts_rank 与最大的余弦相似度（都是绝对可比量）。"""
    max_fts = max((c.fts_score or 0.0) for c in pool) if pool else 0.0
    max_cos = max((c.vector_score or 0.0) for c in pool) if pool else 0.0
    return max_fts, max_cos


def _llm_rerank(llm: LLMClient, question: str, pool: list[Candidate]) -> list[float]:
    """LLM 重排：把候选逐条编号交给模型打相关性分（0-10），按编号回填。

    返回顺序与传入的 pool 一一对应，调用方负责按分数重排。
    编号→分数必须用字典按 id 回填，不能直接按返回数组顺序 zip：
    一旦模型少给或多给一项，分数就会整体错位（实测踩过这个坑）。
    """
    listing = "\n\n".join(f"[{i}] {c.text[:600]}" for i, c in enumerate(pool))
    prompt = (
        f"问题：{question}\n\n下面有 {len(pool)} 个候选片段，"
        "请给每个片段打 0-10 的相关性分（0=完全无关，10=直接回答该问题）。\n"
        '只输出 JSON，形如 {"scores": [{"id": 0, "score": 7}, {"id": 1, "score": 2}]}，'
        "要为每一个编号都给出一项。\n\n" + listing
    )
    data = llm.json([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1500)
    raw = data.get("scores", []) if isinstance(data, dict) else data
    by_id: dict[int, float] = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(pool):
            by_id[idx] = score
    return [by_id.get(i, 0.0) / 10.0 for i in range(len(pool))]


import re  # noqa: E402

CITE_RE = re.compile(r"\[([A-Za-z0-9_:\-#\.]+#\d+)\]")


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for m in CITE_RE.finditer(text):
        cid = m.group(1)
        if cid not in seen:
            seen.append(cid)
    return seen


def citation_hallucination(citations: list[str], candidate_ids: list[str]) -> list[str]:
    """引用幻觉：回答里出现不在本次送回的候选集内的 chunk_id。"""
    allowed = set(candidate_ids)
    return [c for c in citations if c not in allowed]


class CitationTracker:
    """把每次回答、引用、允许的候选集落盘，供人工抽检与判分一致率复核。"""

    def __init__(self, path=None) -> None:
        self.rows: list[dict[str, Any]] = []
        self.path = path

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    def save(self, path) -> None:
        import json as _json
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
        self.path = p

