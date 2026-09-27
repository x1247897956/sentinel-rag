"""指标定义（全部为公式的直译，检索类指标完全确定性、不经 LLM）。

retrieval:
  recall@k        = (1/N) * Σ_i 1[ gold_chunk_ids_i ∩ top_k_i ≠ ∅ ]
  mrr@k           = (1/N) * Σ_i ( 1 / rank_i )，rank_i = gold 首次出现的名次，未命中记 0
  first_hit@1     = (1/N) * Σ_i 1[ rank_i == 1 ]
  hit_rate@k      = 同 recall@k（保留名称以便报告直读）

generation（只在最终配置 D 上评）:
  cite_hit         = (1/M) * Σ_j 1[ 回答的第 j 个引用 ∈ gold_chunk_ids ]，M = 有引用的可答题数
  cite_halluc_rate = 引用了「不在本次候选集内」的 chunk_id 的回答数 / 可答题数
  points_partial   = (1/M) * Σ_j ( 覆盖的要点数 / 要点总数 )
  points_full      = (1/M) * Σ_j 1[ 全部要点被覆盖 ]
  refusal_acc      = (1/U) * Σ 1[ 知识库外题目被正确拒答 ]
  over_refusal     = (1/A) * Σ 1[ 可答题目被拒答 ]
  faithfulness     = 由判分模型给出（报告中标注模型，并抽检一致率）
"""

from __future__ import annotations

from typing import Any, Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    return 1.0 if gold_set & set(ranked[:k]) else 0.0


def first_hit_rank(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> int | None:
    gold_set = set(gold)
    for i, cid in enumerate(ranked[:k], 1):
        if cid in gold_set:
            return i
    return None


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    r = first_hit_rank(ranked, gold, k)
    return 1.0 / r if r else 0.0


def aggregate_retrieval(rows: list[dict], k_values: Sequence[int] = (5, 10)) -> dict[str, float]:
    answerable = [r for r in rows if r["answerable"]]
    n = len(answerable)
    if n == 0:
        return {}
    out: dict[str, float] = {"n_answerable": n}
    for k in k_values:
        out[f"recall@{k}"] = round(sum(recall_at_k(r["retrieved"], r["gold_chunk_ids"], k) for r in answerable) / n, 4)
    out["mrr@10"] = round(sum(reciprocal_rank(r["retrieved"], r["gold_chunk_ids"], 10) for r in answerable) / n, 4)
    out["first_hit@1"] = round(
        sum(1.0 for r in answerable if first_hit_rank(r["retrieved"], r["gold_chunk_ids"], 10) == 1) / n, 4
    )
    return out


def aggregate_retrieval_by_category(rows: list[dict], k: int = 5) -> dict[str, dict]:
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["answerable"]:
            buckets[r["category"]].append(r)
    out = {}
    for cat, items in sorted(buckets.items()):
        n = len(items)
        out[cat] = {
            "n": n,
            f"recall@{k}": round(sum(recall_at_k(i["retrieved"], i["gold_chunk_ids"], k) for i in items) / n, 4),
            "mrr@10": round(sum(reciprocal_rank(i["retrieved"], i["gold_chunk_ids"], 10) for i in items) / n, 4),
        }
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100 * len(s) + 0.5)) - 1))
    return round(float(s[idx]), 1)


def aggregate_system(rows: list[dict]) -> dict[str, Any]:
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    gen_lat = [r["gen_latency_ms"] for r in rows if r.get("gen_latency_ms")]
    retrieval_lat = [r["retrieval_ms"] for r in rows if r.get("retrieval_ms")]
    rerank_lat = [r["rerank_ms"] for r in rows if r.get("rerank_ms")]
    pt = [r.get("prompt_tokens", 0) for r in rows]
    ct = [r.get("completion_tokens", 0) for r in rows]
    return {
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "p50_retrieval_ms": percentile(retrieval_lat, 50),
        "p95_retrieval_ms": percentile(retrieval_lat, 95),
        "p50_rerank_ms": percentile(rerank_lat, 50),
        "p95_rerank_ms": percentile(rerank_lat, 95),
        "p50_gen_ms": percentile(gen_lat, 50),
        "p95_gen_ms": percentile(gen_lat, 95),
        "avg_prompt_tokens": round(sum(pt) / len(pt), 1) if pt else 0,
        "avg_completion_tokens": round(sum(ct) / len(ct), 1) if ct else 0,
        "total_prompt_tokens": sum(pt),
        "total_completion_tokens": sum(ct),
    }
