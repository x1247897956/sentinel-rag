"""扩增回归集：**只新增，绝不改动已冻结的题**。

设计纪律（来自 `docs/design/SentinelRAG-实测数据采集表.md` §3）：
> 回归集一旦冻结就不再为提分改动——改题=自欺；要加只能新增并保留旧版本记录。

因此本脚本：
  1. 原样保留 `eval/dataset/regression_set.jsonl` 已有的每一行（按原始文本续写，不重新序列化）；
  2. 只从**尚未被任何题当作 gold 的文档**里抽新 chunk 命题；
  3. 新题沿用同一套纪律：`gold_points` 必须在原文里**逐字可查**（`verify_gold_points`），
     核对不通过就丢弃；
  4. 新增的「知识库外」题必须**实测**在库内没有依据（走拒答门禁的两条绝对信号）；
  5. 本次新增的题目、丢弃的候选与原因都落到 `eval/dataset/_draft/` 留痕。

用法：
  uv run python scripts/extend_regression.py --dry-run     # 只打印计划，不写文件
  uv run python scripts/extend_regression.py               # 真写（会先备份原文件）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_regression import (  # noqa: E402
    build_multi_questions,
    gen_single,
    sample_concept_chunks,
    sample_cve_chunks,
    sample_multi_hop,
)
from src.config import EVAL_DIR, REFUSAL_MIN_COSINE, REFUSAL_MIN_FTS, get_settings  # noqa: E402
from src.retrieval import storage  # noqa: E402
from src.retrieval.hybrid import Retriever  # noqa: E402
from src.retrieval.models import LLMClient  # noqa: E402

DATASET = EVAL_DIR / "dataset" / "regression_set.jsonl"
DRAFT_DIR = EVAL_DIR / "dataset" / "_draft"

# 新增的「知识库外」候选池：语料是安全域（CVE/GHSA/ATT&CK/OWASP），这些题目确实无依据。
# 只取前 --unanswerable 个**实测通过**的（走与拒答门禁相同的两条绝对信号）。
EXTRA_UNANSWERABLE = [
    ("玻利维亚乌尤尼盐沼的面积大约是多少平方公里？", "地理，与安全语料无关"),
    ("贝多芬第九交响曲的首演是在哪一年？", "音乐史，与安全语料无关"),
    ("咖啡豆的日晒处理法与水洗处理法在风味上有什么差别？", "食品工艺，与安全语料无关"),
    ("国际象棋中王车易位的规则限制有哪些？", "棋类规则，与安全语料无关"),
]


def load_existing() -> tuple[list[str], list[dict]]:
    raw_lines = [ln for ln in DATASET.read_text(encoding="utf-8").split("\n") if ln.strip()]
    return raw_lines, [json.loads(ln) for ln in raw_lines]


def unanswerable_passes(conn, retriever: Retriever, question: str) -> tuple[bool, dict]:
    """用与拒答门禁完全相同的两条绝对信号，验证这道题在库里确实没有依据。"""
    with conn.cursor() as cur:
        fts = retriever._recall_fts(cur, question, 30)
        vec = retriever._recall_vector(cur, question, 30)
    max_fts = max((s for _, s in fts), default=0.0)
    max_cos = max((s for _, s in vec), default=0.0)
    ok = max_fts < REFUSAL_MIN_FTS and max_cos < REFUSAL_MIN_COSINE
    return ok, {"max_ts_rank": round(max_fts, 6), "max_cos": round(max_cos, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", type=int, default=10, help="新增概念题数（目标 25）")
    ap.add_argument("--multi", type=int, default=4, help="新增多跳题数（语料上限 8）")
    ap.add_argument("--unanswerable", type=int, default=2)
    ap.add_argument("--single", type=int, default=0, help="新增单跳题数（默认不再增加）")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_lines, existing = load_existing()
    used_docs = {d for it in existing for d in it.get("gold_doc_ids", [])}
    used_chunks = {c for it in existing for c in it.get("gold_chunk_ids", [])}
    used_questions = {it["question"].strip() for it in existing}
    print(f"[extend] 既有 {len(existing)} 题，已占用 gold 文档 {len(used_docs)} 篇")

    settings = get_settings()
    llm = LLMClient(model=settings.gen_model)
    conn = storage.connect()
    retriever = Retriever(conn, reranker=None)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    new_items: list[dict] = []
    rejected: list[dict] = []

    # ---- 概念题：从没被用过的 OWASP / ATT&CK 文档里抽 ----
    if args.concept:
        pool = [s for s in sample_concept_chunks(conn, 400, seed=args.seed) if s["doc_id"] not in used_docs]
        print(f"[extend] 概念题可用新样本 {len(pool)} 个，申请 {args.concept} 题（多抽 2 倍做缓冲）")
        items, rej = gen_single(conn, llm, pool[: args.concept * 2], "concept")
        rejected += rej
        kept = []
        for it in items:
            if it["question"].strip() in used_questions or it["gold_doc_ids"][0] in used_docs:
                continue
            used_docs.update(it["gold_doc_ids"])
            used_questions.add(it["question"].strip())
            kept.append(it)
            if len(kept) >= args.concept:
                break
        new_items += kept
        print(f"[extend] 概念题新增 {len(kept)}")

    # ---- 多跳题：语料里只有 4 组「同一产品 ≥2 篇不同公告」，上限 8 题 ----
    # 约束只用「两条公告必须不同 + 这对组合没被用过」：已冻结的集合里本来就存在跨题共用
    # gold 文档（q043 与 q072 共用 CVE-2025-71392#2），所以不按"文档从未被用过"来卡，
    # 否则多跳题永远长不大（实测候选组会直接掉到 0）。
    if args.multi:
        multi_doc_pairs = {
            frozenset(it["gold_doc_ids"]) for it in existing if it["category"] == "multi_hop"
        }
        pairs = []
        for p in sample_multi_hop(conn, 50, seed=args.seed):
            a, b = p["items"]
            if frozenset({a["doc_id"], b["doc_id"]}) in multi_doc_pairs:
                continue
            pairs.append(p)
        multi = build_multi_questions(pairs)
        kept = []
        for it in multi:
            if it["question"].strip() in used_questions:
                continue
            used_chunks.update(it["gold_chunk_ids"])
            used_questions.add(it["question"].strip())
            kept.append(it)
            if len(kept) >= args.multi:
                break
        new_items += kept
        print(f"[extend] 多跳题新增 {len(kept)}（候选组 {len(pairs)}，语料上限 8）")

    # ---- 知识库外题：必须实测过不了拒答阈值 ----
    if args.unanswerable:
        kept = []
        for q, why in EXTRA_UNANSWERABLE:
            if q in used_questions:
                continue
            ok, sig = unanswerable_passes(conn, retriever, q)
            print(f"[extend] 知识库外候选「{q[:24]}…」实测 {sig} → {'通过' if ok else '丢弃'}")
            if not ok:
                rejected.append({"question": q, "reason": f"库内似有依据：{sig}"})
                continue
            kept.append(
                {
                    "question": q,
                    "answer": None,
                    "gold_points": [],
                    "gold_doc_ids": [],
                    "gold_chunk_ids": [],
                    "category": "unanswerable",
                    "answerable": False,
                    "difficulty": "easy",
                    "source_type": "none",
                    "note": why,
                }
            )
            used_questions.add(q)
            if len(kept) >= args.unanswerable:
                break
        new_items += kept
        print(f"[extend] 知识库外题新增 {len(kept)}")

    # ---- 单跳题（默认不加，保留开关）----
    if args.single:
        pool = [s for s in sample_cve_chunks(conn, 400, seed=args.seed) if s["doc_id"] not in used_docs]
        items, rej = gen_single(conn, llm, pool[: args.single * 2], "single_hop")
        rejected += rej
        kept = []
        for it in items:
            if it["question"].strip() in used_questions or it["gold_doc_ids"][0] in used_docs:
                continue
            used_docs.update(it["gold_doc_ids"])
            used_questions.add(it["question"].strip())
            kept.append(it)
            if len(kept) >= args.single:
                break
        new_items += kept
        print(f"[extend] 单跳题新增 {len(kept)}")
    conn.close()

    # ---- 追加：原 84 行按原文续写，保证逐字节不变 ----
    start = len(existing) + 1
    for i, it in enumerate(new_items, start):
        it["qid"] = f"q{i:03d}"
        it.setdefault("answerable", True)
    print(f"[extend] 合计新增 {len(new_items)} 题（q{start:03d}~q{start + len(new_items) - 1:03d}）")

    if args.dry_run:
        for it in new_items:
            print(f"  + {it['qid']} [{it['category']}] {it['question'][:70]}")
        print("[extend] --dry-run，未写文件")
        return 0

    if not new_items:
        print("[extend] 没有可新增的题，未改动文件")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = DRAFT_DIR / f"regression_set.{stamp}.bak.jsonl"
    shutil.copy2(DATASET, backup)
    body = "\n".join(raw_lines + [json.dumps(it, ensure_ascii=False) for it in new_items]) + "\n"
    DATASET.write_text(body, encoding="utf-8")
    (DRAFT_DIR / f"extension.{stamp}.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in new_items), encoding="utf-8"
    )
    (DRAFT_DIR / f"extension_rejected.{stamp}.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in rejected), encoding="utf-8"
    )
    print(f"[extend] 原文件备份 → {backup}")
    print(f"[extend] 写出 → {DATASET}（{len(raw_lines)} + {len(new_items)} = {len(raw_lines) + len(new_items)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
