# 架构

> 与 `README.md` 首屏的图一致。真实边界：**单机 / 千级 chunk / 单租户**，离线评测，无线上部署。

```
┌──────────────── 评测与可观测（本项目的重点）───────────────────────┐
│  回归集 85 条 · 检索/生成双维度指标 · Badcase 四分类归因            │
│  runs 表逐问落库（含模型与 prompt 版本）· CI 门禁：指标掉线即 fail   │
└──────────────────────────────┬───────────────────────────────────┘
                               │ 评测结果 / 运行轨迹
┌──────────────── 服务层（FastAPI）────────────────────────────────┐
│  POST /ask（带引用回答）· POST /ingest（增量入库）· GET /health     │
│  GET /runs（延迟与 token 观测）                                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────── 检索层（核心链路）───────────────────────────────┐
│  ① 全文检索：应用层 jieba 预分词 → tsvector + GIN → ts_rank        │
│  ② 向量检索：BGE-small-zh → pgvector HNSW（余弦）                  │
│         └──────────► ③ RRF 融合（只用名次，不用分数）              │
│  ④ 重排：bge-reranker-base cross-encoder（可开关，A/B/C/D 对照）    │
│  ⑤ 引用溯源：chunk_id 回填 + 强制引用 + 引用幻觉校验               │
│  ⑥ 上下文预算裁剪 → 被裁掉的候选计入 context_truncated             │
│  ⑦ 候选最高分 < 阈值 → 拒答（知识库外问题）                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────── 知识层 ──────────────────────────────────────────┐
│  公开语料：CVE List v5 / GHSA(OSV) / MITRE ATT&CK / OWASP         │
│  → 解析 → 按结构分类型分块（版本区间不切散）→ 元数据              │
│  → content_hash 去重 → 增量入库 / stale 失效标记                   │
│  存储：PostgreSQL 16 + pgvector 0.8.6（documents / chunks /        │
│        runs / eval_runs）                                          │
└───────────────────────────────────────────────────────────────────┘
```

## 数据流（一次问答）

1. `POST /ask` → `Retriever.retrieve(question, mode)`
2. 两条召回并行取 top-30：
   - `chunks.tsv @@ to_tsquery('simple', jieba_tokens)` 按 `ts_rank` 排序
   - `embedding <=> query_vec` 走 HNSW 索引按余弦距离排序
3. `rrf_fuse([fts_ids, vector_ids])`：`score = Σ 1/(60 + rank)`
4. `hybrid_rerank` 配置下，对候选池的 30 条做 cross-encoder 打分（**不新增召回**，只重排）
5. 按最终顺序塞进上下文，直到 token 预算上限；被裁掉并记录 `truncated`
6. 候选最高分 `< REFUSAL_MIN_SCORE` → 直接拒答，不调用生成模型
7. 生成时每段上下文前缀 `[chunk_id]`，要求逐条标注引用
8. 生成后**校验引用**：回答里的 `chunk_id` 必须落在本次候选集内，否则记为引用幻觉
9. 落 `runs` 表（问题 / 检索轨迹 / 引用 / 模型版本 / prompt 版本 / 延迟 / token）

## 数据流（评测）

`eval/runner.py` 对回归集逐题跑 A/B/C/D 四种配置：
**A 纯向量 / B 纯全文 / C 混合(RRF) / D 混合+重排**；
检索指标（recall@5、recall@10、MRR@10、首条命中率）在**重排前的候选池顺序**上确定性计算；
生成指标只在最终配置 D 上评，并把答案与判分结果冻结到 `eval/snapshot/`，
让 CI 可以**在没有 API key 的情况下**重算引用命中率 / 引用幻觉率 / 拒答正确率。

## 表结构

| 表 | 作用 | 关键列 |
| --- | --- | --- |
| `documents` | 文档层去重与失效 | `doc_id`、`content_hash`、`stale`、`published_at`、`updated_at` |
| `chunks` | 检索与引用的最小单位 | `chunk_id`、`tokens`、`tsv`（生成列）、`embedding vector(512)`、`section`、`affected_versions` |
| `runs` | 逐问轨迹 | `retrieved jsonb`、`citations jsonb`、`prompt_version`、`embed_model`、`rerank_model`、`latency_ms`、`prompt_tokens` |
| `eval_runs` | 每次评测快照 | `config`、`dataset_sha`、`metrics jsonb`、`attribution jsonb`、`git_sha` |

索引：`tsv` 上的 GIN、`embedding` 上的 HNSW（`vector_cosine_ops`）、`stale` 上的部分索引。
