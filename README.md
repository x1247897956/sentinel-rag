# SentinelRAG —— 安全知识库检索 Agent：混合检索 + chunk 级引用溯源 + 分层评测

> 本仓库是「**Agent 工程三件套**」之一：`sentinel-rag`（**检索正确性**）· `silver-guard`（决策可控性，仓库待开源）· [`dsh-llm-guard`](https://github.com/x1247897956/dsh-llm-guard)（输入输出安全）。

**它解决什么问题**：安全知识（CVE 影响版本、GHSA 修复版本、ATT&CK 技术点、OWASP 防护要点）
散在多个公开来源、且大多是英文；直接问一个大模型，它会给你一个**看起来对但版本号可能编的**答案。
这个项目把公开安全语料建成可增量更新的知识库，用「全文 + 向量」两条召回 + RRF 融合 + 重排做检索，
回答**强制带 chunk 级引用**，知识库里没有依据时**直接拒答**；并用一套可复现的评测回答
「改动之后它到底变好了还是变坏了」。

**真实边界（先说清）**：单机、千级 chunk、单租户、离线评测，**没有线上部署、没有真实用户**。

---

## 核心结果（全部可在 [`docs/eval-report.md`](docs/eval-report.md) 复核）

语料 **540 篇 / 2618 chunk**，来自 4 类公开源（CVE List v5 · GitHub Advisory Database · MITRE ATT&CK · OWASP Cheat Sheets）；
回归集 **84 条**（单跳 55 / 概念 15 / 多跳 4 / 知识库外 10），人工核对 `gold_points`，进 git。

**四组单变量对照**（同一批问题、同一份索引，检索指标确定性计算、不经 LLM）：

| 配置 | recall@5 | recall@10 | MRR@10 | 首条命中率 | 检索 P50 |
| --- | --- | --- | --- | --- | --- |
| **A 纯向量**（bge-m3 + HNSW 余弦） | 0.3108 | 0.3378 | 0.2462 | 0.2027 | 63 ms |
| **B 纯全文**（jieba 预分词 + tsvector + `ts_rank`） | 0.8514 | 0.9324 | 0.7861 | 0.7297 | 8 ms |
| **C 混合**（加权 RRF 融合） | 0.8378 | 0.9324 | 0.6464 | 0.4595 | 61 ms |
| **D 混合 + 重排**（分数融合式重排，最终方案） | **0.8649** | **0.9459** | 0.7345 | **0.6081** | 129 ms |

**生成指标（配置 D，LLM 判分 + 人工抽检）**：

| 指标 | 值 | 说明 |
| --- | --- | --- |
| 引用命中率 | **0.9180** | 回答引用的 chunk 落在 gold 内的比例（确定性计算；有引用的可答题 61 条，命中 56 条） |
| 引用幻觉率 | **0.0000** | 引用了「不在本次候选集内」的 chunk 的回答占比（确定性计算） |
| 拒答正确率 | **1.0000** | 10 条知识库外问题全部正确拒答（确定性计算） |
| 过度拒答率 | **0.0405** | 74 条可答题中 3 条被拒 |
| 要点覆盖率（部分） | 0.7249 | 判分模型 `deepseek-v4-pro`，与生成模型不同源 |
| 要点覆盖率（全覆盖） | 0.6056 | |
| faithfulness | 0.8939 | 判分模型输出 |
| 人工抽检一致率 | 0.9286（13/14） | 抽检 19.7% 复核判分 |

**Badcase 四分类归因**（84 条中失败 9 条）：
`retrieval_miss` 4 · `rerank_misorder` 0 · `context_truncated` 5 · `generation_halluc` 0。

**CI 门禁真的拦过**（可点开复核）：[`eval` run 36291977414](https://github.com/x1247897956/sentinel-rag/actions/runs/36291977414)
（PR [#1](https://github.com/x1247897956/sentinel-rag/pull/1) 是一次**刻意制造的负对照**：把重排短路掉）——
前 14 步建库/入库/四组检索全绿，**只有门禁那一步红**，退出码 1，掉线项
`recall@5` 0.8649→0.8378、`MRR@10` 0.7345→0.6487、首条命中率 0.6081→0.4595。
另有两组本地降级实验（候选池 30→5、关掉拒答阈值）同样返回退出码 1，见
[`docs/eval-report.md` §8](docs/eval-report.md)。

> ⚠️ **一个不利于本项目结论的实测发现，照实写在这里**：
> 在**这份语料**上 **B（纯全文）的 MRR 高于 C（混合）**——因为语料正文以英文为主、
> 问题以中文为主，向量一路的 `recall@5` 只有 0.31，融合时引入了噪声。
> 加权 RRF 与重排都是为了让 D 相对 C 有真实提升（D 的四个指标全部高于 C——`MRR@10` 0.6464 → 0.7345，
> 且 `recall@10` 与 C 基本持平，符合「重排只改顺序、不新增召回」的预期形状），
> 但 `D > B` 在这份语料上只在 recall@5 / recall@10 上成立，**MRR 与首条命中率不成立**（B 更高：0.7861 vs 0.7345）。
> 这不是实现 bug：55 道单跳题问的是 `CVE-XXXX` 编号，而**抽查 20 个 gold chunk，没有一个在正文里出现过自己的编号**，
> 把编号补进正文后余弦也只从 ~0.47 升到 ~0.55，仍不足以翻盘——标识符本身不含语义，
> 稠密向量天然抓不住它。详细分析与成因见 [`docs/eval-report.md` §4.3](docs/eval-report.md)。

---

## 快速开始

```bash
# 1) 起 PostgreSQL 16 + pgvector，并建表
make db-up
cp .env.example .env        # 填 DEEPSEEK_API_KEY（生成与判分用）

# 2) 装依赖（含本地 BGE embedding / 重排模型）
make setup

# 3) 抓公开语料（CVE / GHSA / ATT&CK / OWASP）并建索引
make fetch
make ingest                 # 增量入库：content_hash 去重 + stale 失效标记

# 4) 跑评测：四组对照 + 生成指标 + Badcase 归因
make eval                   # 结果写 eval/results/，与 docs/eval-report.md 一致
make gate                   # 用 eval/baseline.json 做门禁判定（掉线退出码 1）

# 5) 起服务
make serve                  # POST /ask · POST /ingest · GET /health · GET /runs
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"CVE-2026-100599 影响 OpenClaw 的哪些版本？","mode":"hybrid_rerank"}' | python -m json.tool
```

**端到端自检**：`make smoke` 会依次打印库内状态、**分词生效验证**（整句当一个 token 命中 0 行 vs
jieba 预分词命中上千行）、知识库内问答（带引用）、知识库外拒答、四组检索对照、`/health`。

---

## 架构

```
┌──────────────── 评测与可观测（本项目的重点）───────────────────────┐
│  回归集 84 条 · 检索/生成双维度指标 · Badcase 四分类归因            │
│  runs 表逐问落库（含模型与 prompt 版本）· CI 门禁：指标掉线即 fail   │
└──────────────────────────────┬───────────────────────────────────┘
┌──────────────── 服务层（FastAPI）────────────────────────────────┐
│  POST /ask（带引用回答）· POST /ingest（增量入库）· GET /health     │
└──────────────────────────────┬───────────────────────────────────┘
┌──────────────── 检索层（核心链路）───────────────────────────────┐
│  ① 全文检索：应用层 jieba 预分词 → tsvector + GIN → ts_rank        │
│  ② 向量检索：BGE-m3 → pgvector HNSW（余弦）                        │
│         └──────────► ③ 加权 RRF 融合（只用名次，每路一个权重）      │
│  ④ 重排：分数融合式重排（可切换 cross-encoder / LLM）              │
│  ⑤ 引用溯源：chunk_id 回填 + 强制引用 + 引用幻觉校验               │
│  ⑥ 上下文预算裁剪 → 被裁掉的候选计入 context_truncated             │
│  ⑦ 最高分低于绝对阈值 → 拒答（知识库外问题）                       │
└──────────────────────────────┬───────────────────────────────────┘
┌──────────────── 知识层 ──────────────────────────────────────────┐
│  CVE List v5 / GHSA(OSV) / ATT&CK STIX / OWASP Cheat Sheets        │
│  → 按结构分类型分块（版本区间不切散）→ 元数据 → content_hash 去重   │
│  → 增量入库 / stale 失效标记 → PostgreSQL 16 + pgvector 0.8.6      │
└───────────────────────────────────────────────────────────────────┘
```

详见 [`docs/architecture.md`](docs/architecture.md)。

---

## 关键设计取舍（面试官最会追问的部分，完整版见 [`docs/design-notes.md`](docs/design-notes.md)）

- **中文分词放应用层，不装 `zhparser`**：扩展要自编译镜像、本地与容器产物不一致；
  入库前用 `jieba` 预分词存 `tokens` 列，再建 `tsvector` + GIN，检索链路与部署都变确定。
- **融合用 RRF，并给每一路一个名次权重**：余弦相似度与 `ts_rank` 量纲不可比，RRF 只用名次；
  本语料上全文一路明显更可靠，故权重 3:1——权重是**每路一个常数**，不是每条候选一个分数。
- **重排没用 cross-encoder（诚实交代）**：`bge-reranker-base` 在「中文问题 + 英文正文」上
  区分度不足（同一问题下 30 个候选分数几乎全在 0.5~0.62），LLM 重排单问要 2~18s；
  最终用确定性、毫秒级的分数融合式重排，三种后端代码都保留、可切换对比。
- **拒答阈值必须用绝对量**：候选池内归一化会让冠军恒等于 1.0，阈值永远失效
  （本项目踩过这个坑）；改用「最大 `ts_rank` ≥ 0.008 或最大余弦 ≥ 0.62」。
- **分块按结构而不是固定字符数**：CVE 按语义段、ATT&CK 按技术点、OWASP 按标题层级，
  再在段内按「行 → 句 → 字符窗口」打包，**版本区间绝不切散**（有单测守着）。
- **评测分两层**：检索指标全部确定性计算、不经 LLM；生成指标才用判分模型，
  且判分模型与生成模型**不同源**，并做人工抽检报告一致率。

---

## 已知限制 / 未做

- **规模**：单机、千级 chunk（2618）、单租户；十万级以上才需要认真比较向量库分片与 HNSW 参数。
- **向量一路偏弱**：语料以英文为主、问题以中文为主，`bge-m3` 的跨语种检索只做到 `recall@5` 0.31。
  换成更强的多语种/英文域 embedding 是明确的下一步，但**本次没有做**。
- **重排未达预期**：见上「关键取舍」与 `docs/eval-report.md` §4.3。
- **知识库外的拒答靠阈值**：阈值是在回归集上标定的；本轮没有独立的 held-out 集合来选阈值，
  这一点写在 `docs/eval-report.md` §7。
- **未做**：MCP 工具层、多轮会话、护栏、模型微调/LoRA/vLLM、知识图谱、多模态、线上部署。
- **语料会过期**：CVE/公告会被更新或撤回，因此有 `stale` 失效标记；检索默认过滤失效 chunk。

---

## 数据与合规

- 数据来源（只读拉取，遵守各来源许可）：
  [CVE List v5](https://github.com/CVEProject/cvelistV5) ·
  [GitHub Advisory Database](https://github.com/github/advisory-database) ·
  [MITRE ATT&CK（STIX）](https://github.com/mitre-attack/attack-stix-data) ·
  [OWASP Cheat Sheet Series](https://github.com/OWASP/CheatSheetSeries)。
- **不含任何公司内部数据、规则库、客户数据与真实个人数据**；语料全部为公开源。
- 语料自身权利属于各来源，**不适用本仓库的 MIT 许可**；逐项许可见 [`LICENSE`](LICENSE) 末尾。
- 采集快照（篇数 + 整文件 sha256）冻结在 `eval/snapshot/corpus_manifest.json`，
  用 `uv run python scripts/prepare_corpus.py --verify-only` 可校验手上的语料是否与评测报告一致。

## 许可

代码：[MIT](LICENSE)。模型权重（`BAAI/bge-m3`、`BAAI/bge-reranker-base`）遵循各自模型卡许可。
