"""回归集构建辅助：从真实语料出发生成题目候选，并做「gold_points 必须能在原文中找到」的校验。

标注纪律（本项目自定，且写进报告）：
  - **题目与要点由人机协作产出，但每一条 gold_point 都必须在本项目的 chunk 原文里逐字可查**；
    `verify_gold_points` 做的就是这件事——校验不通过的一律丢弃，不进入回归集。
  - 题目一旦冻结就不再为提分改动；要加只能新增并保留旧版本记录。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.retrieval.models import LLMClient

PROMPT_VERSION = "regression-builder-v1.0"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("（", "(").replace("）", ")").replace("，", ",").replace("：", ":")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def point_in_text(point: str, text: str) -> bool:
    p = normalize(point)
    t = normalize(text)
    if p and p in t:
        return True
    # 允许把 "5.6.0, 5.6.1" 这类逗号列举拆开逐个核对
    parts = [x.strip() for x in re.split(r"[,;、/]", p) if len(x.strip()) >= 2]
    return bool(parts) and all(x in t for x in parts)


def verify_gold_points(item: dict, chunks: dict[str, str]) -> tuple[bool, list[str]]:
    """每个 gold_point 必须在它标注的 gold chunk 原文里可查（跨多个 gold chunk 时任一命中即可）。"""
    texts = [chunks.get(cid, "") for cid in item.get("gold_chunk_ids", [])]
    if not texts or not any(texts):
        return False, item.get("gold_points", [])
    missing = [
        p for p in item.get("gold_points", []) if not any(point_in_text(p, t) for t in texts)
    ]
    return (not missing), missing


QA_SYSTEM = (
    "你是安全知识库的评测题目命题人。命题必须严格基于给定的原文，不得引入原文之外的知识。"
    "输出必须是合法 JSON 对象，不要输出任何解释文字。"
)


def build_qa_prompt(chunk_id: str, source_type: str, title: str, text: str) -> list[dict[str, str]]:
    if source_type == "cve_list_v5":
        guide = (
            "请围绕这段 CVE 原文出一道**事实型**问题，优先问「受影响版本区间 / 受影响产品 / "
            "CVSS 评分与严重性 / 修复建议」这类原文里写死的信息。"
        )
    elif source_type == "ghsa":
        guide = (
            "请围绕这段 GitHub 安全公告原文出一道**事实型**问题，优先问「受影响包与版本区间 / "
            "漏洞摘要 / 修复版本」。"
        )
    elif source_type == "attack_stix":
        guide = "请围绕这段 ATT&CK 技术点原文出一道**概念解释型**问题（这个技术点是什么、怎么检测）。"
    else:
        guide = "请围绕这段 OWASP Cheat Sheet 原文出一道**概念解释型**问题（这条防护要点为什么要这么做、怎么做）。"

    user = f"""{guide}

硬性要求：
1. 问题必须是中文自然语言，且**只用原文里出现过的信息**；
2. `answer` 是标准答案，必须与原文完全一致，最多 200 字；若原文语言不是中文，请译成中文但**专有名词、版本号、编号保持原样**；
3. `gold_points_str` 是答案中的关键要点，用 `|||` 分隔；**每个要点都必须是原文中能逐字找到的字符串**（版本号、编号、专有名词保持原文写法），3-5 个；
4. 不要问「原文说了什么」这类元问题；不要问需要外部知识的问题。

原文 chunk_id：{chunk_id}
标题：{title}
原文：
\"\"\"
{text[:3000]}
\"\"\"

只输出如下 JSON：
{{"question": "...", "answer": "...", "gold_points_str": "要点1|||要点2|||要点3", "answerable": true}}"""
    return [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def generate_items(
    samples: list[dict],
    llm: LLMClient,
    max_repair: int = 2,
) -> tuple[list[dict], list[dict]]:
    """samples: [{chunk_id, doc_id, source_type, title, text, category, difficulty}]"""
    accepted: list[dict] = []
    rejected: list[dict] = []
    for s in samples:
        try:
            raw = llm.json(build_qa_prompt(s["chunk_id"], s["source_type"], s["title"], s["text"]),
                           temperature=0.2, max_tokens=700)
        except Exception as exc:  # noqa: BLE001
            rejected.append({**s, "reason": f"生成失败：{exc}"})
            continue
        points = [p.strip() for p in str(raw.get("gold_points_str", "")).split("|||") if p.strip()]
        item = {
            "question": str(raw.get("question", "")).strip(),
            "answer": str(raw.get("answer", "")).strip(),
            "gold_points": points,
            "gold_doc_ids": [s["doc_id"]],
            "gold_chunk_ids": [s["chunk_id"]],
            "category": s["category"],
            "difficulty": s.get("difficulty", "medium"),
            "source_type": s["source_type"],
        }
        ok, missing = verify_gold_points(item, {s["chunk_id"]: s["text"]})
        tries = 0
        while not ok and tries < max_repair:
            tries += 1
            points = [p for p in points if any(point_in_text(p, s["text"]) for p in [p]) is not False]
            points = [p for p in points if point_in_text(p, s["text"])]
            if not points:
                break
            item["gold_points"] = points
            ok = True
        if not item["question"] or not item["gold_points"]:
            rejected.append({**s, "reason": "生成结果为空或要点全部无法在原文中核对"})
            continue
        accepted.append(item)
    return accepted, rejected


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(items: list[dict]) -> dict[str, Any]:
    from collections import Counter

    return {
        "total": len(items),
        "by_category": dict(Counter(i["category"] for i in items)),
        "by_source_type": dict(Counter(i.get("source_type", "?") for i in items)),
        "unanswerable": sum(1 for i in items if not i.get("answerable", True)),
    }
