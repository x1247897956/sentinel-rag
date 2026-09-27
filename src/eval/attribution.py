"""Badcase 四分类归因。

分类顺序即优先级（与设计文档 §4.3 一致）：
  ① retrieval_miss      gold 完全没进候选池 —— 分块 / 分词 / embedding 的问题
  ② rerank_misorder     召回了但被挤出送入上下文的名次 —— 重排模型 / 候选数问题
  ③ context_truncated   进了候选但被 token 预算裁掉 —— 预算 / 冗余 chunk 问题
  ④ generation_halluc   上下文里有答案却答错或编造 —— prompt / 生成模型问题
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ["retrieval_miss", "rerank_misorder", "context_truncated", "generation_halluc"]


def classify(item: dict, trace_dict: dict, gen: dict | None) -> str | None:
    if not item.get("answerable", True):
        # 知识库外题目：唯一正确的行为是拒答
        return None if (gen or {}).get("refused") else "generation_halluc"

    gold = set(item.get("gold_chunk_ids", []))
    retrieved = trace_dict.get("candidates", [])
    retrieved_ids = [c["chunk_id"] for c in retrieved]
    if not (gold & set(retrieved_ids)):
        return "retrieval_miss"

    context_ids = set(trace_dict.get("context", []))
    if gold & context_ids:
        # 上下文里有 gold：若要点没覆盖 → 生成问题
        if gen and gen.get("points_covered") is not None and gen.get("points_covered") < 1.0:
            return "generation_halluc"
        if gen and gen.get("citations") and not (set(gen["citations"]) & gold):
            return "generation_halluc"
        return None

    truncated = set(trace_dict.get("truncated", []))
    if gold & truncated:
        return "context_truncated"
    return "rerank_misorder"


def summarize_badcases(rows: list[dict]) -> dict[str, Any]:
    from collections import Counter, defaultdict

    counter: Counter[str] = Counter()
    cases: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bc = r.get("badcase")
        if not bc:
            continue
        counter[bc] += 1
        cases[bc].append(
            {
                "qid": r["qid"],
                "question": r["question"],
                "category": r["category"],
                "gold_chunk_ids": r["gold_chunk_ids"],
                "top5": r["retrieved"][:5],
                "top_score": r.get("top_score"),
                "refused": r.get("refused"),
            }
        )
    total_failed = sum(counter.values())
    return {
        "total_cases": len(rows),
        "failed": total_failed,
        "by_category": {c: counter.get(c, 0) for c in CATEGORIES},
        "cases": {c: cases.get(c, []) for c in CATEGORIES},
    }


def markdown_report(attribution: dict[str, Any], limit_per_category: int = 5) -> str:
    lines = [
        "## 5. 失败案例分析（Badcase 四分类归因）",
        "",
        f"评测集 {attribution['total_cases']} 条，失败 {attribution['failed']} 条（最终配置 D）。",
        "",
        "| 归因类别 | 含义 | 条数 |",
        "| --- | --- | --- |",
        f"| `retrieval_miss` | gold 完全没进候选池（分块 / 分词 / embedding） | {attribution['by_category']['retrieval_miss']} |",
        f"| `rerank_misorder` | 召回了但被挤出送入上下文的名次（重排 / 候选数） | {attribution['by_category']['rerank_misorder']} |",
        f"| `context_truncated` | 进了候选但被 token 预算裁掉（预算 / 冗余 chunk） | {attribution['by_category']['context_truncated']} |",
        f"| `generation_halluc` | 上下文里有答案却答错或编造（prompt / 生成模型） | {attribution['by_category']['generation_halluc']} |",
        "",
    ]
    for cat, cases in attribution["cases"].items():
        if not cases:
            continue
        lines.append(f"### {cat}（{len(cases)} 条，示例最多 {limit_per_category} 条）")
        lines.append("")
        for c in cases[:limit_per_category]:
            lines.append(
                f"- `{c['qid']}` {c['question']} ｜ gold={c['gold_chunk_ids']} ｜ top5={c['top5']}"
                f" ｜ top_score={c['top_score']} ｜ refused={c['refused']}"
            )
        lines.append("")
    return "\n".join(lines)
