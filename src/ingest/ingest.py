"""入库：分块 → 批量编码 embedding → 写 PostgreSQL（去重 + 增量 + stale）。

用法：
  uv run python -m src.ingest.ingest                 # 增量入库
  uv run python -m src.ingest.ingest --all           # 全量（先清空）
  uv run python -m src.ingest.ingest --limit 50      # 冒烟测试
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.config import (  # noqa: F401  (REPO_ROOT_STR 供日志)
    CORPUS_DIR,
    EMBED_MODEL,
    REPO_ROOT_STR,
    ROOT,
)
from src.ingest.pipeline import build_chunks, chunk_stats, load_corpus
from src.retrieval import storage
from src.retrieval.models import Embedder

CACHE_DIR = ROOT / "data" / "cache"


def ingest(all_docs: bool = False, limit: int | None = None, batch: int = 32, dump: Path | None = None) -> dict:
    t0 = time.time()
    corpus = load_corpus()
    if limit:
        corpus = corpus[:limit]
    print(f"[ingest] 语料 {len(corpus)} 篇")

    # embedding 缓存键只用「模型 + 语料文档集合」：与 chunker 版本、库里已有数据无关，
    # 因此「全量重建」也能命中缓存（CI 每次都是干净库，这是唯一能省下全量编码的办法）。
    key_raw = EMBED_MODEL + "|" + "|".join(sorted(d.to_record()["content_hash"] for d in corpus))
    embed_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]

    conn = storage.connect()
    # 先按 content_hash 判重：内容没变的文档**根本不进编码阶段**——
    # 否则一次「没有变化」的增量入库仍要付全量 embedding 的钱（实测 2619 chunk 约 350s）。
    if all_docs:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE runs, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()
    pending: list = []
    skipped_unchanged = 0
    with conn.cursor() as cur:
        for d in corpus:
            cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (d.doc_id,))
            row = cur.fetchone()
            if row is not None and row[0] == d.to_record()["content_hash"]:
                skipped_unchanged += 1
                continue
            pending.append(d)
    if skipped_unchanged:
        print(f"[ingest] 内容未变、跳过编码的文档：{skipped_unchanged} 篇")

    docs, chunks = build_chunks(pending)
    stats = chunk_stats(chunks)
    print(f"[ingest] 待处理语料 {len(docs)} 篇，分块 {stats}")

    embed_s = 0.0
    vectors: list[list[float]] = []
    if chunks:
        # embedding 缓存：语料/模型没变就复用向量文件。改了 chunker 会让行数不匹配，
        # 那时自动回退到重新编码（不会静默用错向量）。
        inputs = [f"{c.section} {c.text}" if c.section else c.text for c in chunks]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"embeddings_{embed_key}.jsonl"
        if cache_file.exists():
            print(f"[ingest] embedding 缓存命中：{cache_file.name}")
            with open(cache_file, encoding="utf-8") as fh:
                vectors = [json.loads(line) for line in fh if line.strip()]
            if len(vectors) != len(chunks):
                print("[ingest] 缓存行数与 chunk 数不一致，忽略缓存并重新编码")
                vectors = []
        if not vectors:
            print(f"[ingest] 开始编码 embedding（{EMBED_MODEL}）")
            embedder = Embedder()
            t_embed = time.time()
            vectors = embedder.encode(inputs, batch_size=batch)
            embed_s = time.time() - t_embed
            print(f"[ingest] embedding 完成，{embed_s:.1f}s（{len(chunks) / max(embed_s, 1e-6):.0f} chunk/s）")
            with open(cache_file, "w", encoding="utf-8") as fh:
                for v in vectors:
                    fh.write(json.dumps([round(x, 6) for x in v]) + "\n")
            print(f"[ingest] embedding 缓存写出 → {cache_file.name}")
    counters = {
        "inserted_docs": 0,
        "unchanged_docs": skipped_unchanged,
        "updated_docs": 0,
        "inserted_chunks": 0,
        "skipped_chunks": 0,
    }
    try:
        with conn.cursor() as cur:
            by_doc: dict[str, list[int]] = {}
            for i, c in enumerate(chunks):
                by_doc.setdefault(c.doc_id, []).append(i)
            for n, doc in enumerate(docs, 1):
                rec = doc.to_record()
                action = storage.upsert_document(cur, rec)
                counters[f"{action}_docs" if action != "unchanged" else "unchanged_docs"] += 1
                if action == "unchanged":
                    counters["skipped_chunks"] += len(by_doc.get(doc.doc_id, []))
                    continue
                idxs = by_doc.get(doc.doc_id, [])
                n_ins = storage.insert_chunks(cur, rec, [chunks[i] for i in idxs], [vectors[i] for i in idxs])
                counters["inserted_chunks"] += n_ins
                counters["skipped_chunks"] += len(idxs) - n_ins
                if n % 100 == 0:
                    conn.commit()
                    print(f"[ingest] {n}/{len(docs)} 篇，已入库 chunk {counters['inserted_chunks']}")
            conn.commit()
            with conn.cursor() as c2:
                db_stats = storage.stats(c2)
            if dump:
                n = storage.dump_chunks_jsonl(c2 if False else cur, dump)
                print(f"[ingest] chunk 快照写出 {n} 行 → {dump}")
    finally:
        conn.close()

    result = {
        "documents": len(docs),
        "chunks": len(chunks),
        "chunk_stats": stats,
        "embed_seconds": round(embed_s, 1),
        "embed_model": EMBED_MODEL,
        "counters": counters,
        "db": db_stats,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "db"}, ensure_ascii=False, indent=2))
    print("[ingest] 库内状态：" + json.dumps(db_stats, ensure_ascii=False))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="清空后全量重建")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dump", type=Path, default=None, help="把入库后的 chunk 快照写成 JSONL")
    args = ap.parse_args()
    ingest(all_docs=args.all, limit=args.limit, batch=args.batch, dump=args.dump)
