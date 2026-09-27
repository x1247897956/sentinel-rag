"""单元测试：不依赖数据库与网络的确定性部分。"""

from __future__ import annotations

import math

from src.eval.attribution import classify
from src.eval.metrics import aggregate_retrieval, first_hit_rank, recall_at_k, reciprocal_rank
from src.ingest.chunker import _estimate_tokens, chunk_document, tokenize
from src.ingest.schema import Doc, sha256_text
from src.retrieval.hybrid import (
    CitationTracker,
    citation_hallucination,
    extract_citations,
    rrf_fuse,
)


def make_cve_doc() -> Doc:
    text = "\n\n".join(
        [
            "## 漏洞描述\n某组件存在命令注入漏洞，攻击者可远程执行任意代码。",
            "## 受影响产品\n- vendor product",
            "## 受影响版本区间\nversion=5.6.0 < 5.6.3 [:affected]\nversion=5.4.0 [:unaffected]",
            "## 严重性\nCVSS 基础评分 9.8，严重性等级 critical",
        ]
    )
    return Doc(
        doc_id="CVE-TEST-0001",
        source="test",
        source_type="cve_list_v5",
        title="CVE-TEST-0001 漏洞公告",
        text=text,
        cve_id="CVE-TEST-0001",
        severity="critical",
        cvss=9.8,
        affected_versions="version=5.6.0 < 5.6.3",
    )


def test_version_range_not_split():
    """版本区间不能被切散：同一区间的两行必须落在同一个 chunk 里。"""
    chunks = chunk_document(make_cve_doc(), target=420, overlap=80)
    hits = [c for c in chunks if "version=5.6.0 < 5.6.3" in c.text]
    assert hits, "版本区间段落应在某个 chunk 中"
    assert "version=5.4.0" in hits[0].text, "同一版本区间的多行不得被拆到不同 chunk"
    assert "5.6" in hits[0].tokens and "3" in hits[0].tokens  # jieba 会把 5.6.3 切成 5.6 / 3


def test_chunk_ids_are_citable():
    chunks = chunk_document(make_cve_doc())
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"CVE-TEST-0001#{i}"
        assert c.content_hash == sha256_text(c.text.strip())


def test_tokenizer_is_applied_to_chinese():
    tokens = tokenize("跨版本可分析性增强")
    assert " " in tokens, "中文必须被切成多个 token，否则 to_tsvector 形同废掉"
    assert len(tokens.split()) >= 2


def test_rrf_fusion_ranks_agreeing_docs_higher():
    a = ["c1", "c2", "c3"]
    b = ["c2", "c1", "c4"]
    fused = rrf_fuse([a, b])
    assert fused["c1"] > fused["c3"]
    assert fused["c2"] > fused["c4"]
    assert math.isclose(fused["c2"], 1 / 62 + 1 / 61, rel_tol=1e-9)


def test_rrf_weight_shifts_preference():
    """加权 RRF：权重高的一路能把「只在另一路靠前」的候选压下去。"""
    weak = ["x1", "x2"]
    strong = ["y1", "y2", "y3", "y4", "y5"]
    fused = rrf_fuse([(weak, 1.0), (strong, 3.0)])
    assert fused["y5"] > fused["x1"], "3/(60+5) > 1/(60+1)，说明权重按预期生效"
    assert fused["y1"] > fused["x1"]


def test_recall_and_mrr():
    ranked = ["x", "y", "g1", "g2"]
    assert recall_at_k(ranked, ["g1"], 5) == 1.0
    assert recall_at_k(ranked, ["zzz"], 5) == 0.0
    assert first_hit_rank(ranked, ["g1"]) == 3
    assert math.isclose(reciprocal_rank(ranked, ["g1"]), 1 / 3)
    assert reciprocal_rank(ranked, ["zzz"]) == 0.0


def test_aggregate_retrieval_matches_hand_computation():
    rows = [
        {"answerable": True, "category": "single_hop", "retrieved": ["a", "g1"], "gold_chunk_ids": ["g1"]},
        {"answerable": True, "category": "single_hop", "retrieved": ["a", "b"], "gold_chunk_ids": ["g2"]},
        {"answerable": False, "category": "unanswerable", "retrieved": [], "gold_chunk_ids": []},
    ]
    agg = aggregate_retrieval(rows, k_values=(1, 5))
    assert agg["n_answerable"] == 2
    assert agg["recall@1"] == 0.0
    assert agg["recall@5"] == 0.5
    assert math.isclose(agg["mrr@10"], (1 / 2 + 0) / 2, rel_tol=1e-9)
    assert agg["first_hit@1"] == 0.0


def test_citations_extraction_and_hallucination():
    text = "结论一 [CVE-2024-3094#1]，结论二 [CVE-2024-3094#1]，编造 [XX-1#9]。"
    cites = extract_citations(text)
    assert cites == ["CVE-2024-3094#1", "XX-1#9"]
    assert citation_hallucination(cites, ["CVE-2024-3094#1", "CVE-2024-3094#2"]) == ["XX-1#9"]
    assert citation_hallucination(cites, ["CVE-2024-3094#1", "XX-1#9"]) == []


def test_badcase_four_way_classification():
    item = {"answerable": True, "gold_chunk_ids": ["d#1"], "gold_points": ["p"]}
    # ① 完全没召回
    t = {"candidates": [{"chunk_id": "x#1"}], "context": ["x#1"], "truncated": []}
    assert classify(item, t, None) == "retrieval_miss"
    # ② 召回了但没进上下文（被重排挤出）
    t = {"candidates": [{"chunk_id": "d#1"}, {"chunk_id": "x#1"}], "context": ["x#1"], "truncated": []}
    assert classify(item, t, None) == "rerank_misorder"
    # ③ 进了候选但被 token 预算裁掉
    t = {
        "candidates": [{"chunk_id": "x#1"}, {"chunk_id": "d#1"}],
        "context": ["x#1"],
        "truncated": ["d#1"],
    }
    assert classify(item, t, None) == "context_truncated"
    # ④ 上下文里有答案但生成没覆盖要点
    t = {"candidates": [{"chunk_id": "d#1"}], "context": ["d#1"], "truncated": []}
    assert classify(item, t, {"points_covered": 0.0, "citations": ["d#1"]}) == "generation_halluc"
    # 答对 → 无 badcase
    assert classify(item, t, {"points_covered": 1.0, "citations": ["d#1"]}) is None
    # 知识库外题目：答了就是幻觉类失败
    unans = {"answerable": False, "gold_chunk_ids": [], "gold_points": []}
    assert classify(unans, t, {"refused": False}) == "generation_halluc"
    assert classify(unans, t, {"refused": True}) is None


def test_token_budget_estimate_is_monotonic():
    assert _estimate_tokens("中文中文") < _estimate_tokens("中文中文中文中文")


def test_citation_tracker_roundtrip(tmp_path):
    tr = CitationTracker()
    tr.add(config="D", question="q", answer="a", citations=["c#1"], allowed=["c#1"], top_chunk="c#1")
    out = tmp_path / "answers.jsonl"
    tr.save(out)
    assert out.read_text(encoding="utf-8").count("\n") == 1
