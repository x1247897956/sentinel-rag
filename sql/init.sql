-- SentinelRAG schema (PostgreSQL 16 + pgvector)
-- 设计依据：docs/design（内部材料，不随仓库公开）方案 §2.4 / §5

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- 文档层：去重、增量与失效标记的最小单位 ----------
CREATE TABLE IF NOT EXISTS documents (
  doc_id        TEXT PRIMARY KEY,          -- CVE-2024-3094 / GHSA-xxxx / T1059 / owasp:xxx
  source        TEXT NOT NULL,             -- 来源仓库或站点
  source_type   TEXT NOT NULL,             -- cve_list_v5 / ghsa / attack_stix / owasp_cheatsheet
  title         TEXT,
  published_at  TIMESTAMPTZ,
  updated_at    TIMESTAMPTZ,
  content_hash  TEXT NOT NULL,             -- 整篇规范化文本的 sha256（去重 / 增量判据）
  fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  stale         BOOLEAN NOT NULL DEFAULT FALSE,
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ---------- 分块层：检索与引用溯源的最小单位 ----------
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      TEXT PRIMARY KEY,          -- f"{doc_id}#{idx}"
  doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  idx           INT  NOT NULL,
  text          TEXT NOT NULL,
  tokens        TEXT NOT NULL,             -- jieba 预分词结果（空格连接），供 tsvector 使用
  tsv           tsvector GENERATED ALWAYS AS (to_tsvector('simple', tokens)) STORED,
  embedding     vector(1024),              -- BAAI/bge-m3 = 1024 维（多语种，中文查询 + 英文语料）
  source        TEXT,
  source_type   TEXT,
  cve_id        TEXT,
  severity      TEXT,
  cvss          REAL,
  published_at  TIMESTAMPTZ,
  updated_at    TIMESTAMPTZ,
  affected_versions TEXT,
  section       TEXT,
  content_hash  TEXT NOT NULL,             -- chunk 文本 sha256
  embed_model   TEXT,                      -- 记录 embedding 版本，保证评测可复现
  stale         BOOLEAN NOT NULL DEFAULT FALSE,
  UNIQUE (doc_id, content_hash)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin     ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx     ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_live        ON chunks (stale) WHERE stale = FALSE;

-- HNSW 索引：pgvector 0.5+ 支持；维度固定 1024（与 embedding 模型绑定）
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

-- ---------- 运行轨迹（可观测） ----------
CREATE TABLE IF NOT EXISTS runs (
  run_id            UUID PRIMARY KEY,
  qid               TEXT,
  question          TEXT NOT NULL,
  config            TEXT,                  -- A/B/C/D 配置名
  retrieved         JSONB,                 -- [{chunk_id, rrf_score, rerank_score, rank_each}]
  answer            TEXT,
  citations         JSONB,
  prompt_version    TEXT,
  embed_model       TEXT,
  rerank_model      TEXT,
  gen_model         TEXT,
  latency_ms        INT,
  retrieval_ms      INT,
  rerank_ms         INT,
  prompt_tokens     INT,
  completion_tokens INT,
  refused           BOOLEAN DEFAULT FALSE,
  badcase           TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS runs_created_at ON runs (created_at DESC);

-- ---------- 评测结果（每次 run 一份快照，供门禁比较） ----------
CREATE TABLE IF NOT EXISTS eval_runs (
  eval_id       UUID PRIMARY KEY,
  config        TEXT NOT NULL,
  dataset_sha   TEXT NOT NULL,
  metrics       JSONB NOT NULL,
  attribution   JSONB NOT NULL,
  git_sha       TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
