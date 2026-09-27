"""构建回归集：从库内真实 chunk 分层抽样 → 生成题目 → 校验 gold_points 可在原文中逐字核对。

用法：
  uv run python -m scripts.build_regression --single 55 --concept 20 --multi 15 --unanswerable 10

标注说明（写进报告）：
  - 题目与标准答案由 LLM 基于**指定 chunk 原文**起草，用于规模化起草；
  - **每一条 gold_point 都必须在该 chunk 原文里逐字可查**，否则丢弃该条
    （`verify_gold_points`）——gold 不允许是"模型认为对"的表述；
  - 人工抽查核对，抽样结果记录在 docs/eval-report.md。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVAL_DIR, get_settings  # noqa: E402
from src.eval.regression import (  # noqa: E402
    build_qa_prompt,
    point_in_text,
    summarize,
    verify_gold_points,
    write_jsonl,
)
from src.retrieval import storage  # noqa: E402
from src.retrieval.models import LLMClient  # noqa: E402

# 知识库外题目：语料是安全域（CVE/GHSA/ATT&CK/OWASP），这些问题在库里确实没有依据
UNANSWERABLE = [
    ("深海热液喷口附近的管虫靠什么共生菌获得能量？", "生物学，与安全语料无关"),
    ("《红楼梦》里贾府一共经历了多少次元宵节？", "文学，与安全语料无关"),
    ("2024 年诺贝尔生理学或医学奖颁给了哪两位科学家？", "时事奖项，与安全语料无关"),
    ("三体问题在数值计算中有哪些经典积分方法？", "天体力学，与安全语料无关"),
    ("宋代汝窑天青釉的烧制温度区间是多少？", "陶瓷工艺史，与安全语料无关"),
    ("北京地铁 10 号线早高峰的最小发车间隔是多少秒？", "城市交通运营，与安全语料无关"),
    ("马拉松运动员比赛日碳水摄入的推荐克数是多少？", "运动营养，与安全语料无关"),
    ("如何在家庭阳台上种植矮生番茄并提高坐果率？", "园艺，与安全语料无关"),
    ("钢琴调律时标准音 A4 的频率是多少赫兹？", "音乐声学，与安全语料无关"),
    ("南美安第斯山脉的的的喀喀湖最大水深是多少米？", "地理，与安全语料无关"),
]


def sample_cve_chunks(conn, n: int, seed: int = 42) -> list[dict]:
    """按信息量优先抽样 CVE chunk：优先「受影响版本区间」「严重性」「修复建议」等有事实的段落。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.text, c.section, c.source_type, c.severity, c.cvss,
                   d.title
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.stale = FALSE AND c.source_type = 'cve_list_v5'
              AND c.section IN ('受影响版本区间', '严重性', '修复建议', '漏洞描述')
              AND length(c.text) BETWEEN 120 AND 2200
            ORDER BY c.doc_id DESC
            """
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    # 每题只用一篇文档的一个 chunk 作 gold（避免同一 gold 重复计入多题）
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    section_rank = {"受影响版本区间": 0, "严重性": 1, "修复建议": 2, "漏洞描述": 3}
    picked: list[dict] = []
    rnd = random.Random(seed)
    docs = list(by_doc)
    rnd.shuffle(docs)
    for doc_id in docs:
        cands = sorted(by_doc[doc_id], key=lambda r: (section_rank.get(r["section"], 9), len(r["text"])))
        picked.append(cands[0])
        if len(picked) >= n:
            break
    return picked


def sample_concept_chunks(conn, n: int, seed: int = 7) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.text, c.section, c.source_type, c.severity, c.cvss, d.title
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.stale = FALSE AND c.source_type IN ('owasp_cheatsheet', 'attack_stix')
              AND length(c.text) BETWEEN 300 AND 2600
            ORDER BY c.doc_id
            """
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    rnd = random.Random(seed)
    docs = list(by_doc)
    rnd.shuffle(docs)
    out = []
    for doc_id in docs:
        cands = sorted(by_doc[doc_id], key=lambda r: -len(r["text"]))
        out.append(cands[0])
        if len(out) >= n:
            break
    return out


def sample_multi_hop(conn, n: int, seed: int = 11) -> list[dict]:
    """多跳/条件筛选：取同一产品下带版本区间的多篇公告，问题需要跨文档筛选。

    这里只产出「样本组」，问题文本由 build_multi_questions 用模板 + 原文事实生成，
    保证 gold_doc_ids 与 gold_chunk_ids 与题目严格对应。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.text, c.section, c.source_type, d.title, d.meta
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.stale = FALSE AND c.source_type = 'cve_list_v5'
              AND c.section = '受影响版本区间' AND length(c.text) BETWEEN 120 AND 2000
            ORDER BY c.doc_id
            """
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    groups: dict[str, list[dict]] = {}
    for r in rows:
        meta = r["meta"] or {}
        products = meta.get("products") or []
        key = products[0] if products else None
        if not key:
            continue
        groups.setdefault(key, []).append(r)
    # 选产品维度上有多篇公告的组，随机取 2 篇构成一道多跳题
    rnd = random.Random(seed)
    keys = [k for k, v in groups.items() if len(v) >= 2]
    rnd.shuffle(keys)
    out = []
    used_chunks: set[str] = set()
    for k in keys:
        items = [x for x in groups[k] if x["chunk_id"] not in used_chunks]
        if len(items) < 2:
            continue
        # 必须来自**不同文档**，否则题目会退化成"同一个 CVE 问两遍"（已修此 bug）
        by_doc: dict[str, dict] = {}
        for x in items:
            by_doc.setdefault(x["doc_id"], x)
        # 每个产品内**反复取互不重复的对**，直到该产品的公告用完或凑够 n 题。
        # 早先这里每个产品只 append 一次就换下一个产品，于是 n 再大也只能得到
        # 「产品组数」道题（实测 4 组产品 → 恒为 4 题，--multi 15 完全不起作用）。
        docs = list(by_doc.values())
        rnd.shuffle(docs)
        while len(docs) >= 2 and len(out) < n:
            a, b = docs.pop(), docs.pop()
            used_chunks.update({a["chunk_id"], b["chunk_id"]})
            out.append({"product": k, "items": [a, b]})
        if len(out) >= n:
            break
    return out


def gen_single(conn, llm: LLMClient, samples: list[dict], category: str) -> tuple[list[dict], list[dict]]:
    items: list[dict] = []
    rejected: list[dict] = []
    for i, s in enumerate(samples, 1):
        try:
            raw = llm.json(
                build_qa_prompt(s["chunk_id"], s["source_type"], s["title"], s["text"]),
                temperature=0.2,
                max_tokens=800,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(samples)}] 生成失败：{exc}")
            rejected.append({**s, "reason": str(exc)})
            continue
        points = [p.strip() for p in str(raw.get("gold_points_str", "")).split("|||") if p.strip()]
        item = {
            "question": str(raw.get("question", "")).strip(),
            "answer": str(raw.get("answer", "")).strip(),
            "gold_points": points,
            "gold_doc_ids": [s["doc_id"]],
            "gold_chunk_ids": [s["chunk_id"]],
            "category": category,
            "difficulty": "easy" if category == "single_hop" else "medium",
            "source_type": s["source_type"],
        }
        ok, missing = verify_gold_points(item, {s["chunk_id"]: s["text"]})
        if not ok:
            # 只保留能在原文逐字核对的要点；残留不足 2 个要点则丢弃整条
            kept = [p for p in points if point_in_text(p, s["text"])]
            if len(kept) < 2 or not item["question"]:
                rejected.append({**s, "reason": f"要点无法在原文核对：{missing[:3]}"})
                continue
            item["gold_points"] = kept
        if not item["question"]:
            rejected.append({**s, "reason": "问题为空"})
            continue
        items.append(item)
        print(f"  [{i}/{len(samples)}] ok: {item['question'][:50]}")
    return items, rejected


def build_multi_questions(pairs: list[dict]) -> list[dict]:
    """多跳题用模板生成，要点全部取自两篇原文的版本区间文本，保证可核对。"""
    items: list[dict] = []
    ver_re = re.compile(r"(?:version[=<]+\s*[\w\.\-\*]+|<=\s*[\w\.\-\*]+|<\s*[\w\.\-\*]+)")
    for p in pairs:
        a, b = p["items"]

        def versions(item):
            vs = []
            for line in item["text"].split("\n"):
                m = ver_re.search(line)
                if m:
                    vs.append(m.group(0).strip())
                if len(vs) >= 2:
                    break
            return vs

        va, vb = versions(a), versions(b)
        if not va or not vb:
            continue
        q = f"{p['product']} 下，{a['doc_id']} 与 {b['doc_id']} 两个漏洞各自的受影响版本区间分别是什么？"
        ans = (
            f"{a['doc_id']} 的受影响版本区间包含 {'、'.join(va)}；"
            f"{b['doc_id']} 的受影响版本区间包含 {'、'.join(vb)}。"
        )
        items.append(
            {
                "question": q,
                "answer": ans,
                "gold_points": [f"{a['doc_id']}", f"{b['doc_id']}"] + va + vb,
                "gold_doc_ids": [a["doc_id"], b["doc_id"]],
                "gold_chunk_ids": [a["chunk_id"], b["chunk_id"]],
                "category": "multi_hop",
                "difficulty": "hard",
                "source_type": "cve_list_v5",
            }
        )
    verified = []
    for it in items:
        chunks = " ".join([it["answer"]])
        if all(point_in_text(pt, chunks) for pt in it["gold_points"]):
            verified.append(it)
    return verified


def build_unanswerable() -> list[dict]:
    return [
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
        for q, why in UNANSWERABLE
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=int, default=55)
    ap.add_argument("--concept", type=int, default=20)
    ap.add_argument("--multi", type=int, default=15)
    ap.add_argument("--unanswerable", type=int, default=10)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "dataset" / "regression_set.jsonl")
    ap.add_argument("--draft-dir", type=Path, default=EVAL_DIR / "dataset" / "_draft")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    settings = get_settings()
    llm = LLMClient(model=settings.gen_model)
    conn = storage.connect()
    args.draft_dir.mkdir(parents=True, exist_ok=True)

    print("== 抽样 ==")
    cve_samples = sample_cve_chunks(conn, args.single, seed=args.seed)
    concept_samples = sample_concept_chunks(conn, args.concept, seed=args.seed)
    multi_pairs = sample_multi_hop(conn, args.multi, seed=args.seed)
    print(f"单跳样本 {len(cve_samples)}｜概念样本 {len(concept_samples)}｜多跳组 {len(multi_pairs)}")

    print("== 生成单跳题 ==")
    single_items, rej1 = gen_single(conn, llm, cve_samples, "single_hop")
    print("== 生成概念题 ==")
    concept_items, rej2 = gen_single(conn, llm, concept_samples, "concept")
    write_jsonl(args.draft_dir / "single_hop.jsonl", single_items)
    write_jsonl(args.draft_dir / "concept.jsonl", concept_items)

    print("== 组装多跳题 ==")
    multi_items = build_multi_questions(multi_pairs)
    write_jsonl(args.draft_dir / "multi_hop.jsonl", multi_items)

    unans = build_unanswerable()[: args.unanswerable]
    write_jsonl(args.draft_dir / "unanswerable.jsonl", unans)

    all_items = single_items + multi_items + concept_items + unans
    for i, it in enumerate(all_items, 1):
        it["qid"] = f"q{i:03d}"
        it.setdefault("answerable", True)
    write_jsonl(args.out, all_items)
    print("== 汇总 ==")
    print(json.dumps(summarize(all_items), ensure_ascii=False, indent=2))
    print("被丢弃的候选：", len(rej1) + len(rej2))
    write_jsonl(args.draft_dir / "rejected.jsonl", rej1 + rej2)
    print(f"写出 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
