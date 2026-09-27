"""存储层：PostgreSQL + pgvector。

- 全文检索：对应用层 jieba 预分词后的 tokens 建 tsvector + GIN 索引，查询走 ts_rank
  （这是 `ts_rank`，不是 BM25——`ts_rank` 不含文档长度归一化与 tf 饱和）。
- 向量检索：pgvector HNSW + 余弦距离，embedding 已归一化。
- 增量入库：documents.content_hash 判重；内容变化 → 旧文档 stale = TRUE。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import psycopg
from psycopg.types.json import Jsonb

from src.config import EMBED_DIM, EMBED_MODEL, get_settings


def connect(dsn: str | None = None) -> psycopg.Connection:
    dsn = dsn or get_settings().database_url
    conn = psycopg.connect(dsn, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public")
    return conn


def _vec_literal(vec: Sequence[float]) -> str:
    if len(vec) != EMBED_DIM:
        raise ValueError(f"embedding 维度必须为 {EMBED_DIM}，收到 {len(vec)}")
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


# ---------------------------------------------------------------- 写入


def upsert_document(cur: psycopg.Cursor, doc: dict) -> str:
    """返回 'inserted' | 'unchanged' | 'updated'。

    - 未见过 → insert
    - content_hash 相同 → unchanged（增量入库跳过，不重复计费/重建索引）
    - content_hash 不同 → 更新文档 + 旧的 chunk 打 stale（保留以便复现历史评测）
    """
    cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (doc["doc_id"],))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            """
            INSERT INTO documents (doc_id, source, source_type, title, published_at, updated_at,
                                   content_hash, meta, stale)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            """,
            (
                doc["doc_id"],
                doc["source"],
                doc["source_type"],
                doc["title"],
                doc.get("published_at"),
                doc.get("updated_at"),
                doc["content_hash"],
                Jsonb(doc.get("meta") or {}),
            ),
        )
        return "inserted"
    if row[0] == doc["content_hash"]:
        return "unchanged"
    cur.execute(
        "UPDATE chunks SET stale = TRUE WHERE doc_id = %s AND stale = FALSE", (doc["doc_id"],)
    )
    cur.execute(
        """
        UPDATE documents SET source = %s, source_type = %s, title = %s, published_at = %s,
               updated_at = %s, content_hash = %s, meta = %s, fetched_at = now(), stale = FALSE
        WHERE doc_id = %s
        """,
        (
            doc["source"],
            doc["source_type"],
            doc["title"],
            doc.get("published_at"),
            doc.get("updated_at"),
            doc["content_hash"],
            Jsonb(doc.get("meta") or {}),
            doc["doc_id"],
        ),
    )
    return "updated"


def insert_chunks(cur: psycopg.Cursor, doc: dict, chunks: Iterable[Any], embeddings: list[list[float]] | None = None) -> int:
    """按内容 hash 去重后写入 chunk。返回实际新增行数。"""
    inserted = 0
    for i, ch in enumerate(chunks):
        cur.execute(
            "SELECT 1 FROM chunks WHERE doc_id = %s AND content_hash = %s",
            (ch.doc_id, ch.content_hash),
        )
        if cur.fetchone() is not None:
            continue
        emb = _vec_literal(embeddings[i]) if embeddings else None
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, idx, text, tokens, embedding, source, source_type,
                                cve_id, severity, cvss, published_at, updated_at, affected_versions,
                                section, content_hash, embed_model, stale)
            VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (
                ch.chunk_id,
                ch.doc_id,
                ch.idx,
                ch.text,
                ch.tokens,
                emb,
                doc["source"],
                doc["source_type"],
                doc.get("cve_id"),
                doc.get("severity"),
                doc.get("cvss"),
                doc.get("published_at"),
                doc.get("updated_at"),
                doc.get("affected_versions"),
                ch.section,
                ch.content_hash,
                # 必须写实际使用的模型名（EMBED_MODEL），否则评测口径与库内 provenance 不一致
                EMBED_MODEL if emb else None,
            ),
        )
        inserted += cur.rowcount
    return inserted


def mark_doc_stale(cur: psycopg.Cursor, doc_id: str) -> int:
    cur.execute("UPDATE documents SET stale = TRUE WHERE doc_id = %s", (doc_id,))
    cur.execute("UPDATE chunks SET stale = TRUE WHERE doc_id = %s", (doc_id,))
    return cur.rowcount


# ---------------------------------------------------------------- 检索


def build_tsquery(query_tokens: str) -> str:
    """把 jieba 分词结果转成安全的 tsquery 字符串。

    分词结果里会带 `.` `(` `)` `？` 这类标点（例如 "v0.9"、"get_device_capability()"），
    直接拼进 to_tsquery 会被当成操作符而报语法错误，因此先做字符级清洗。
    这里用 `|`（OR）而不是 `&`：安全语料里一句话往往只有一个关键词是关键标识符，
    OR 保证召回；排序质量由 ts_rank 与后续 RRF/重排负责。
    """
    import re as _re

    cleaned: list[str] = []
    for tok in query_tokens.split():
        safe = _re.sub(r"[^\w\u4e00-\u9fff]+", "", tok, flags=_re.UNICODE)
        if len(safe) >= 1:
            cleaned.append(safe)
    if not cleaned:
        return ""
    return " | ".join(dict.fromkeys(cleaned))


def fts_search(cur: psycopg.Cursor, query_tokens: str, limit: int = 30) -> list[tuple[str, float]]:
    """全文检索：ts_rank over 预分词 tsvector。返回 [(chunk_id, ts_rank)]。"""
    tsquery = build_tsquery(query_tokens)
    if not tsquery:
        return []
    cur.execute(
        """
        SELECT chunk_id, ts_rank(tsv, to_tsquery('simple', %s)) AS rank
        FROM chunks
        WHERE tsv @@ to_tsquery('simple', %s) AND stale = FALSE
        ORDER BY rank DESC, chunk_id
        LIMIT %s
        """,
        (tsquery, tsquery, limit),
    )
    return [(r[0], float(r[1])) for r in cur.fetchall()]


def vector_search(cur: psycopg.Cursor, embedding: Sequence[float], limit: int = 30) -> list[tuple[str, float]]:
    """向量检索：HNSW 余弦距离，返回 [(chunk_id, cosine_similarity)]。"""
    vec = _vec_literal(embedding)
    cur.execute(
        """
        SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL AND stale = FALSE
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec, vec, limit),
    )
    return [(r[0], float(r[1])) for r in cur.fetchall()]


def fetch_chunks(cur: psycopg.Cursor, chunk_ids: Sequence[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    cur.execute(
        """
        SELECT chunk_id, doc_id, idx, text, source, source_type, section, cve_id, severity, cvss,
               published_at, stale, affected_versions
        FROM chunks WHERE chunk_id = ANY(%s)
        """,
        (list(chunk_ids),),
    )
    cols = [d.name for d in cur.description]
    return {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}


def fts_probe(cur: psycopg.Cursor, raw_query: str) -> dict:
    """演示「分词前 vs 分词后」差异：不做分词时整句被当成一个 token，命中 0 行。

    naive 一列模拟的是「把整句原样交给全文检索」的行为；
    jieba 一列是应用层预分词后的真实行为。
    """
    import re as _re

    naive_single_token = _re.sub(r"[^\w\u4e00-\u9fff]+", "", raw_query, flags=_re.UNICODE)
    cur.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('simple', %s) AND stale = FALSE",
        (naive_single_token,),
    )
    naive = cur.fetchone()[0]
    from src.ingest.chunker import tokenize

    toks = tokenize(raw_query)
    tsquery = build_tsquery(toks)
    cur.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('simple', %s) AND stale = FALSE",
        (tsquery,),
    )
    return {
        "naive_whole_sentence_hits": naive,
        "naive_token": naive_single_token,
        "jieba_tokenized_hits": cur.fetchone()[0],
        "tokens": toks,
    }


def stats(cur: psycopg.Cursor) -> dict:
    cur.execute("SELECT count(*) FROM documents")
    docs = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM documents WHERE stale")
    stale_docs = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM chunks")
    chunks = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM chunks WHERE stale")
    stale_chunks = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
    embedded = cur.fetchone()[0]
    cur.execute(
        "SELECT source_type, count(*) FROM documents GROUP BY source_type ORDER BY count(*) DESC"
    )
    by_source = dict(cur.fetchall())
    cur.execute(
        """
        SELECT source_type, count(*) FROM chunks GROUP BY source_type ORDER BY count(*) DESC
        """
    )
    chunks_by_source = dict(cur.fetchall())
    return {
        "documents": docs,
        "documents_stale": stale_docs,
        "chunks": chunks,
        "chunks_stale": stale_chunks,
        "chunks_embedded": embedded,
        "documents_by_source": by_source,
        "chunks_by_source": chunks_by_source,
    }


def log_run(cur: psycopg.Cursor, run: dict) -> None:
    cur.execute(
        """
        INSERT INTO runs (run_id, qid, question, config, retrieved, answer, citations,
                          prompt_version, embed_model, rerank_model, gen_model, latency_ms,
                          retrieval_ms, rerank_ms, prompt_tokens, completion_tokens, refused, badcase)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run["run_id"],
            run.get("qid"),
            run["question"],
            run.get("config"),
            Jsonb(run.get("retrieved") or []),
            run.get("answer"),
            Jsonb(run.get("citations") or []),
            run.get("prompt_version"),
            run.get("embed_model"),
            run.get("rerank_model"),
            run.get("gen_model"),
            run.get("latency_ms"),
            run.get("retrieval_ms"),
            run.get("rerank_ms"),
            run.get("prompt_tokens"),
            run.get("completion_tokens"),
            run.get("refused"),
            run.get("badcase"),
        ),
    )


def log_eval_run(cur: psycopg.Cursor, payload: dict) -> None:
    cur.execute(
        """
        INSERT INTO eval_runs (eval_id, config, dataset_sha, metrics, attribution, git_sha)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            payload["eval_id"],
            payload["config"],
            payload["dataset_sha"],
            Jsonb(payload["metrics"]),
            Jsonb(payload["attribution"]),
            payload.get("git_sha"),
        ),
    )


def load_corpus_records(cur: psycopg.Cursor) -> list[dict]:
    cur.execute("SELECT doc_id, source_type, title, text FROM documents ORDER BY doc_id")
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def dump_chunks_jsonl(cur: psycopg.Cursor, path) -> int:
    cur.execute(
        """
        SELECT chunk_id, doc_id, idx, text, source, source_type, section, cve_id, severity, cvss,
               affected_versions, published_at
        FROM chunks WHERE stale = FALSE ORDER BY chunk_id
        """
    )
    cols = [d.name for d in cur.description]
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            for k, v in list(rec.items()):
                if hasattr(v, "isoformat"):
                    rec[k] = v.isoformat()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
