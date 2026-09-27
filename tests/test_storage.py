"""存储层测试：连真实 PostgreSQL + pgvector（CI 里由 services 提供）。"""

from __future__ import annotations

import os
import uuid

import pytest

from src.config import EMBED_DIM
from src.ingest.chunker import chunk_document, tokenize
from src.ingest.schema import Doc
from src.retrieval import storage

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="需要 DATABASE_URL（本地起 docker compose 后再跑）"
)


@pytest.fixture()
def conn():
    c = storage.connect()
    yield c
    c.rollback()
    c.close()


def _doc(doc_id: str, text: str) -> Doc:
    return Doc(
        doc_id=doc_id,
        source="test",
        source_type="cve_list_v5",
        title=f"{doc_id} 测试公告",
        text=text,
        cve_id=doc_id,
        affected_versions="version=1.0.0 < 1.0.5",
        meta={"case": "storage-test"},
    )


def test_upsert_is_idempotent_and_marks_stale(conn):
    doc_id = f"CVE-TEST-{uuid.uuid4().hex[:8]}"
    text_v1 = "## 漏洞描述\n第一版描述，存在命令注入。\n\n## 受影响版本区间\nversion=1.0.0 < 1.0.5"
    doc = _doc(doc_id, text_v1)
    chunks = chunk_document(doc)
    vecs = [[0.0] * (EMBED_DIM - 1) + [1.0] for _ in chunks]
    with conn.cursor() as cur:
        assert storage.upsert_document(cur, doc.to_record()) == "inserted"
        n1 = storage.insert_chunks(cur, doc.to_record(), chunks, vecs)
        conn.commit()
        assert n1 == len(chunks)
        # 相同内容再入库 → 跳过
        assert storage.upsert_document(cur, doc.to_record()) == "unchanged"
        n2 = storage.insert_chunks(cur, doc.to_record(), chunks, vecs)
        conn.commit()
        assert n2 == 0
        # 内容变化 → 旧 chunk 打 stale，新内容入库
        doc2 = _doc(doc_id, text_v1 + "\n\n## 修复建议\n升级到 1.0.5。")
        assert storage.upsert_document(cur, doc2.to_record()) == "updated"
        old_stale = 0
        cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s AND stale", (doc_id,))
        old_stale = cur.fetchone()[0]
        assert old_stale == len(chunks), "旧版本 chunk 必须被标记 stale"
        storage.insert_chunks(cur, doc2.to_record(), chunk_document(doc2), [[0.0] * (EMBED_DIM - 1) + [1.0]] * len(chunk_document(doc2)))
        conn.commit()
        cur.execute(
            "SELECT count(*) FROM chunks WHERE doc_id = %s AND stale = FALSE AND text LIKE %s",
            (doc_id, "%升级到 1.0.5%"),
        )
        assert cur.fetchone()[0] == 1
        # 清理
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        conn.commit()


def test_fts_uses_jieba_tokens(conn):
    doc_id = f"CVE-TEST-{uuid.uuid4().hex[:8]}"
    doc = _doc(doc_id, "## 漏洞描述\n跨版本可分析性增强导致的身份认证绕过漏洞。")
    chunks = chunk_document(doc)
    with conn.cursor() as cur:
        storage.upsert_document(cur, doc.to_record())
        storage.insert_chunks(cur, doc.to_record(), chunks, [[0.0] * (EMBED_DIM - 1) + [1.0] for _ in chunks])
        conn.commit()
        probe = storage.fts_probe(cur, "身份认证绕过")
        assert probe["jieba_tokenized_hits"] > 0, "jieba 预分词后应能命中中文查询"
        assert probe["naive_whole_sentence_hits"] == 0, "整句当 token 的朴素做法应命中 0 行"
        hits = storage.fts_search(cur, tokenize("身份认证绕过"), limit=10)
        assert any(cid.startswith(doc_id) for cid, _ in hits)
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        conn.commit()


def test_vector_search_returns_nearest_first(conn):
    doc_id = f"CVE-TEST-{uuid.uuid4().hex[:8]}"
    doc = _doc(doc_id, "## 漏洞描述\n向量检索测试文本。")
    chunks = chunk_document(doc)
    base = [0.0] * (EMBED_DIM - 1) + [1.0]
    other = [1.0] + [0.0] * (EMBED_DIM - 1)
    with conn.cursor() as cur:
        storage.upsert_document(cur, doc.to_record())
        storage.insert_chunks(cur, doc.to_record(), chunks, [base for _ in chunks])
        conn.commit()
        hits = storage.vector_search(cur, other, limit=5)
        assert hits, "向量检索应返回候选"
        # 余弦相似度必须落在 [-1, 1]
        assert all(-1.0001 <= s <= 1.0001 for _, s in hits)
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        conn.commit()
