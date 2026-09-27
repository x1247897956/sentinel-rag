# SentinelRAG 评测报告

> **本文件是本项目所有数字的唯一出处。** 每个数字都由本仓库的代码在下面记录的环境里真实跑出来；
> 未实测的项一律写「未测」，不做估算、不做外推。
> 生成时间：2026-09-27（run tag `v4`）
>
> **复现性声明**：本轮在重跑时发现并修掉了一个会让同一命令产出不同数字的缺陷
> —— embedding 缓存曾把向量 `round(x, 6)` 落盘，而入库时 `_vec_literal` 按 `%.7f` 格式化，
> 于是「缓存命中」与「缓存未命中」写进库的向量差在第 6/7 位小数，近似的 HNSW 因此把个别
> 近似并列的候选排出不同顺序（实测纯向量 `recall@5` 0.3108 vs 0.2973）。
> 修法是把无损向量写进缓存、把量化点收敛到 `_vec_literal` 一处（`src/ingest/ingest.py`）。
> 修后验证：**冷缓存全量重建与热缓存重建的索引逐位相同**
> （`md5(string_agg(chunk_id || embedding::text))` = `bd89bb7f58dd310be3f6913e21357e2b` 两次一致），
> 且**两次独立检索评测给出完全相同的四组指标**（见 §9 复现）。本页数字全部取自修后这一版。

---

## 1. 环境与版本（可复现的前提）

| 项 | 值 |
| --- | --- |
| 机器 | Apple Silicon macOS（darwin arm64），CPU 推理，未用 GPU；单机本地 |
| Python | 3.11.16（uv 管理） |
| 数据库 | PostgreSQL 16 + **pgvector 0.8.6**（`docker compose`，镜像 `pgvector/pgvector:pg16`） |
| embedding 模型 | `BAAI/bge-m3`（**revision `5617a9f61b028005a4858fdac845db406aefb181`**，1024 维，CPU） |
| 重排 | **不是 cross-encoder**：`score_fusion`（两路原始分数按每路权重加权，见 §4.3）；`bge-reranker-base` 与 LLM 重排都在代码里、都测过，结论见 §4.3 |
| 生成模型 | `deepseek-chat` → API 返回的 `model` 字段实测为 **`deepseek-flash`**（DeepSeek OpenAI 兼容接口） |
| 判分模型 | `deepseek-v4-pro`（**与生成模型不同源**；实测返回 `model` 字段即 `deepseek-v4-pro`） |
| prompt 版本 | `answer-zh-v1.0`（`src/eval/prompts.py`，改动即需提版本号） |
| 数据集 | `eval/dataset/regression_set.jsonl`，**84 条**，`sha256 = dfcd615a2fd617ae4f5f0d20888ab8ac6c399c2f0f124a6d808ed917c5b3030a` |
| 语料快照 | `eval/snapshot/corpus_manifest.json`（篇数 + 整文件 sha256），540 篇 |
| 延迟是否含 API 网络往返 | **是**（端到端延迟含 DeepSeek 网络往返；检索/重排延迟不含） |
| 成本单价基准日 | 2026-09-27（DeepSeek 公开定价；只用于换算，未做真实账单核对） |

**这次评测的确切命令**（`make eval` 的展开）：

```bash
uv run python -m src.eval.runner \
  --configs vector fts hybrid hybrid_rerank \
  --judge --tag v4
```

重跑时必须先保证库内是同一份索引：`make db-up && make fetch && make ingest`，
然后用 `uv run python scripts/prepare_corpus.py --verify-only` 校验语料 sha256。

---

## 2. 数据集

| 集合 | 条数 | 构成 | 标注方式 |
| --- | --- | --- | --- |
| 回归集（合计） | **84** | 单跳事实 55 · 概念解释 15 · 半结构化/多跳 4 · 知识库外 10 | 见下 |
| 可答题 | 74 | — | `gold_chunk_ids` + `gold_points` |
| 知识库外 | 10 | 生物/文学/地理/交通/营养/园艺/音乐等与安全语料无关的问题 | `answerable=false` |

**标注纪律（本项目自定，写在 `scripts/build_regression.py` 里）**

1. 题目与标准答案由 LLM 基于**指定的单个 chunk 原文**起草；
2. **每一条 `gold_point` 都必须在该 chunk 原文里逐字可查**，校验不通过就丢弃该条
   （`verify_gold_points`）——不允许出现「模型认为对」的表述；本轮起草 90 条，因要点无法核对丢弃 6 条；
3. 多跳题由模板生成，要点直接取自两篇原文的版本区间文本（修过一个「同一 CVE 问两遍」的生成 bug）；
4. 知识库外的 10 条**人工挑选**，且已逐条确认库内没有对应内容。

语料构成（`wc -l data/corpus/*.jsonl` 的真实输出）：

| 来源 | 语料文件行数 | 入库文档数 | 来源许可 |
| --- | --- | --- | --- |
| CVE List v5 | 320 | 320 | CVE Program Terms of Use |
| GitHub Advisory Database（OSV） | 120 | 120 | CC-BY-4.0 |
| MITRE ATT&CK（STIX） | 60 | 60 | MITRE ATT&CK Terms of Use |
| OWASP Cheat Sheet Series | 40 | 40 | CC-BY-SA-4.0 |
| **合计** | **540** | **540** | — |

---

## 3. 指标定义（先写公式，再写数字）

### 3.1 检索指标（**确定性计算，不经 LLM**）

设第 $i$ 题的 gold chunk 集合为 $G_i$，检索返回的有序列表为 $R_i$（$R_i^{(k)}$ 表示前 k 个）。

| 指标 | 公式 |
| --- | --- |
| `recall@k` | $\frac{1}{N}\sum_{i} \mathbb{1}\left[G_i \cap R_i^{(k)} \neq \varnothing\right]$ |
| `MRR@10` | $\frac{1}{N}\sum_{i} \frac{1}{\mathrm{rank}_i}$，$\mathrm{rank}_i$ 为 gold 首次出现的名次，未进前 10 记 0 |
| 首条命中率 | $\frac{1}{N}\sum_{i} \mathbb{1}\left[\mathrm{rank}_i = 1\right]$ |

$N$ = 74（可答题；知识库外题目不进检索指标）。

**一个必须说清的口径**：配置 D 的检索指标算在**重排后的顺序**上。
重排的作用就是改变顺序，若仍用重排前的顺序算，就永远看不到重排的作用（本项目先踩过这个坑）。
配置 C 没有重排，用的是融合顺序。

### 3.2 生成指标（配置 D）

| 指标 | 公式 | 是否经 LLM |
| --- | --- | --- |
| 引用命中率 | $\frac{1}{M}\sum_j \mathbb{1}\left[C_j \cap G_j \neq \varnothing\right]$，$M$=有引用的可答题数（**61**） | 否（确定性） |
| 引用幻觉率 | $\frac{1}{N}\sum_j \mathbb{1}\left[\exists c \in C_j : c \notin \text{候选集}_j\right]$ | 否（确定性） |
| 拒答正确率 | $\frac{1}{U}\sum \mathbb{1}[\text{知识库外题目被拒答}]$，$U$=10 | 否（确定性） |
| 过度拒答率 | $\frac{1}{N}\sum \mathbb{1}[\text{可答题被拒答}]$ | 否（确定性） |
| 要点覆盖率（部分） | $\frac{1}{M'}\sum_j \frac{\text{覆盖的 gold\_points 数}}{\text{gold\_points 总数}}$ | **是**（`deepseek-v4-pro`） |
| 要点全覆盖率 | $\frac{1}{M'}\sum_j \mathbb{1}[\text{全部要点被覆盖}]$ | **是** |
| faithfulness | 判分模型对「是否只依据给定片段」的判定比例 | **是** |

$C_j$ = 回答里出现的 chunk_id 集合；$M'$ = 有判分结论的可答题数（71）。
两个分母不同：$M=61$ 是**回答里带引用**的条数（74 条可答题中 3 条被拒、10 条回答未带引用），
$M'=71$ 是**被判分模型打过分**的条数（含未带引用但已作答的条目）。

### 3.3 LLM 判分的诚实处理

- 判分模型 `deepseek-v4-pro` 与生成模型 `deepseek-flash` **不同源**；
- prompt 固定版本（`judge-zh-v1.0`），每个 case 输出判分理由；
- **人工抽检 14/71 = 19.7%**，分档一致 **13/14 = 0.9286**；唯一不一致：
  `q044` 判分 0.0 / 人工 1.0（回答把 5 条版本区间以「2.2.0 至 2.8.4」的形式给出，
  与 gold 的 `>= 2.2.0, <= 2.8.4` 语义等价但字面不同 → **判分模型偏严**，
  且这一条的 `reason` 字段为空，判分模型没有给出理由）。抽检明细见 `reports/human_audit.csv`。

---

## 4. 四组对照实验

### 4.1 结果表

同一批 84 条问题、同一份索引（2618 chunk）、同一套参数，只改检索配置：

| 配置 | recall@5 | recall@10 | MRR@10 | 首条命中率 | 检索 P50 | 检索 P95 |
| --- | --- | --- | --- | --- | --- | --- |
| **A 纯向量**（bge-m3 + HNSW 余弦） | 0.3108 | 0.3378 | 0.2462 | 0.2027 | 63 ms | 83 ms |
| **B 纯全文**（jieba 预分词 → tsvector + `ts_rank`） | 0.8514 | 0.9324 | **0.7861** | **0.7297** | 4 ms | 8 ms |
| **C 混合**（加权 RRF，权重 向量:全文 = 1:3） | 0.8378 | 0.9324 | 0.6464 | 0.4595 | 61 ms | 79 ms |
| **D 混合 + 重排**（分数融合式重排，最终方案） | **0.8649** | **0.9459** | 0.7345 | 0.6081 | 120 ms | 177 ms |

**D 相对 C 的增量**：`recall@5 +2.7pt`、`recall@10 +1.4pt`、`MRR@10 +8.8pt`、`首条命中率 +14.9pt`。
`recall@10` 基本持平、MRR 与首条命中率明显提升，符合「重排只对同一候选池重新排序、不新增召回」的预期。

### 4.2 题型分层（用来验证两条召回各自在工作）

| 查询类型 | 纯向量 A | 纯全文 B | 混合 C | 混合+重排 D |
| --- | --- | --- | --- | --- |
| 单跳事实（含 CVE 编号 / 版本号，n=55） | 0.1636 | **0.9636** | 0.9091 | 0.9455 |
| 概念解释（n=15） | **0.8667** | 0.6667 | 0.7333 | 0.7333 |
| 多跳 / 条件筛选（n=4） | 0.25 | 0.00 | 0.25 | 0.25 |

（表中为 `recall@5`。）

**这张表比总表更能说明问题**：
- 含精确标识符的单跳题上，**全文检索碾压向量**（0.96 vs 0.16）——这正是「CVE 编号 + 版本号」这类查询的特点；
- 概念解释题上**向量反超全文**（0.87 vs 0.67）——语义近似、同义改写是它的强项；
- 两条召回确实在各自的题型上更强，所以「B 与 A 没有差异」这个「分词没生效」的告警不成立；
- 多跳题四组都很差（0.00~0.25），是当前最明确的短板（见 §5 与 §7）。

### 4.3 必须主动交代的一件事：**在这份语料上 D 并没有打赢 B**

形状要求是 `D > C > max(A,B)`。实测：

- `D > C`：**成立**（四个指标全部高于 C）；
- `C > A`：成立；
- **`D > B` 不成立**：`recall@5` / `recall@10` 上 D 略高于 B（0.8649 vs 0.8514 / 0.9459 vs 0.9324），
  但 **`MRR@10` 与首条命中率上 B 更高**（0.7861 vs 0.7345 / 0.7297 vs 0.6081）。

**成因（用实测数据说话，不是猜）**：
1. 语料正文以**英文**为主（CVE/GHSA/ATT&CK），问题以**中文**为主；
2. 纯向量一路的 `recall@5` 只有 **0.3108**，全文一路是 **0.8514**；
   等权 RRF 会让「向量排第 1、但全文完全没命中」的噪声候选（1/61）压过
   「全文排第 25」的真实命中（1/85）——这是 C 低于 B 的直接原因；
3. 因此把 RRF 改成**加权**（全文权重 ×3），同批对照实测（`--configs hybrid`，只改 `RRF_WEIGHTS`）：

   | RRF 权重（向量:全文） | recall@5 | recall@10 | MRR@10 | 首条命中率 |
   | --- | --- | --- | --- | --- |
   | 1:1（等权） | 0.8378 | 0.9054 | 0.6110 | 0.4324 |
   | **1:3（采用）** | 0.8378 | **0.9324** | **0.6464** | **0.4595** |

   加权把 `MRR@10` 从 0.6110 提到 0.6464、`recall@10` 从 0.9054 提到 0.9324
   （`recall@5` 不变），但仍不足以超过 B；
4. 重排让 D 相对 C 明显变好，却没有把「混合引入的噪声」完全补回来。

**为什么这次没有去「修」向量一路（一次实测，否掉了一个看似合理的改法）**：
55 道单跳题的题面都带 `CVE-XXXXXXX` 这类精确标识符，而**抽查前 20 个单跳 gold chunk，
0 个在自己的正文里出现过自己的 `doc_id`**（编号只在 `#0` 分段里）。
一个自然的想法是「把 doc_id / 标题拼进每个 chunk 再编码」，于是实测了这件事：

| | 中文问题 vs 英文 gold 正文的余弦 |
| --- | --- |
| 现状（正文只有描述） | 0.429 ~ 0.854（中位数 ≈ 0.50） |
| 拼接 `doc_id + 标题` 后 | 0.513 ~ 0.799（中位数 ≈ 0.58，多数 +0.06~0.10） |

提升真实存在但**量级不够**：`CVE-2025-70150` 这种编号对稠密模型是**无数语义的串**，
补进正文只能把余弦抬 ~0.07，不足以把它推进前 5。
因此本轮没有为此重建索引（重建会让全部数字与快照失效，而收益不足以改变结论）；
`§5.1` 的 `q059` 与 `§5.2` 的 `q053` 属于同一根因。

**重排后端的选择过程（三种都实现并测过）**：

| 后端 | 实测行为 | 结论 |
| --- | --- | --- |
| `cross_encoder`（`bge-reranker-base`，max_length=256/默认都试过） | 同一问题下 30 个候选的分数几乎全在 0.5~0.62，排序与融合结果脱节；CPU 上 30 候选约 9 s | 弃用：该权重对「中文问题 + 英文正文」区分度不足 |
| `llm`（DeepSeek 逐条打分） | 打分可用但单问 2~18 s；打分偏低时会把本可回答的问题判成拒答（触发拒答阈值） | 弃用：成本与延迟不划算，且与生成同源 |
| `score_fusion`（两路**原始**分数按每路权重加权） | 确定性、毫秒级（P50 2 ms） | **采用** |

> 代码里三个后端都保留、可切换（`--rerank-backend`），因此这个结论是可复核的，不是一次性实验。

### 4.4 分词生效验证（确定性）

`GET /health` 之外，`uv run python -m src.cli probe "<中文查询>"` 会给出：

| 做法 | 命中行数（示例查询） |
| --- | --- |
| 整句当成一个 token（不切词） | **0** |
| 应用层 jieba 预分词（实际实现） | **1507** |

（查询：“OneFlow v0.9 中 flow.cuda.get_device_capability() 组件的漏洞是什么？”；库内 2618 chunk）

---

## 5. 失败案例分析（**比数字值钱**）

配置 D 下 84 条中失败 **9 条**：

| 归因类别 | 含义 | 条数 |
| --- | --- | --- |
| `retrieval_miss` | gold 完全没进候选池（分块 / 分词 / embedding） | 4 |
| `rerank_misorder` | 召回了但被挤出送入上下文的名次 | 0 |
| `context_truncated` | 进了候选但被 token 预算裁掉 | 5 |
| `generation_halluc` | 上下文里有答案却答错或编造 | 0 |

完整清单（含 top5 候选与 top_score）在 `eval/results/badcase_v3.md`。

### 5.1 一个讲完整的 case：`q059`（T1547.001）

```
问题：T1547.001 这个技术点是什么？它可以通过哪些启动文件夹路径和注册表键实现持久化？
gold：T1547.001#1
检索 top5：T1037.003#0, T1037.003#1, T1003.008#0, T1003.008#1, T1003.008#2   ← gold 根本没进候选池
最高分：0.560     拒答：是
归因：retrieval_miss
```

**为什么失败**：这道题问的是 ATT&CK 的启动项持久化技术，但返回的全是**其它技术点**——
`ts_rank` 这一路被「启动」「持久化」等泛化词带偏，向量这一路对英文技术点描述也不敏感；
gold chunk（`T1547.001#1`）本身没有出现查询里的任何强标识符（编号只在 `#0` 里），
因此两条召回都没把它排进前 30。

**怎么改**：把 ATT&CK 的「技术点编号」显式写进每个 chunk 的 tokens（而不是只写进标题），
让 `T1547.001` 这种编号在每个子 chunk 里都可被精确命中；同时把「概念题」的召回池扩大。
**本轮没有做这个改动**——它属于 §7 记录的已知偏差，不做「改完再测直到好看」的动作。

### 5.2 另一类：`context_truncated`（5 条）

5 条里有 3 条是多跳题（`q072`/`q073`/`q074`），gold 是两篇公告的「受影响版本区间」chunk，
它们**进了候选池但没进最终 6 段上下文**——因为排在前面的同类候选太多（都是各 CVE 的版本区间），
预算被先到者占满。这说明候选去冗余（同一 doc 只留 1 段、按题面里的编号优先）是明确的下一步。

---

## 6. 生成指标（配置 D）

| 指标 | 值 | 说明 |
| --- | --- | --- |
| 可答题 / 知识库外 | 74 / 10 | — |
| **引用命中率** | **0.9180** | 有引用的可答题 61 条，命中 56 条 |
| **引用幻觉率** | **0.0000** | 0 条引用了候选集之外的 chunk_id |
| 无引用率 | 0.1351 | 10/74 的回答没带 `[chunk_id]`（多为拒答式回答或片段不足的说明） |
| **拒答正确率** | **1.0000** | 10 条知识库外问题全部拒答 |
| 过度拒答率 | 0.0405 | 74 条可答题中 3 条被拒（`q053`/`q059` 两条检索未命中 + `q064`） |
| 要点覆盖率（部分） | 0.7249 | 判分模型 `deepseek-v4-pro` |
| 要点全覆盖率 | 0.6056 | 天然偏低：多数题的 `gold_points` 有 3~5 条 |
| faithfulness | 0.8939 | 判分模型输出 |
| 人工抽检一致率 | **0.9286**（13/14） | 抽检 19.7%，唯一分歧见 §3.3 |

**拒答阈值是怎么定的（诚实说明）**：用两个**绝对可比量**——候选池内最大 `ts_rank ≥ 0.008`
或最大余弦相似度 `≥ 0.62`。这两个阈值是在**这 84 条回归集**上标定的，
本轮**没有独立的 held-out 集合**来做阈值选择，因此存在「阈值对这批题过拟合」的风险（见 §10）。
标定时的分布依据（在当前这份索引上重测）：

| 分组 | n | `max_ts_rank` 中位数 | `max_ts_rank` 最大值 | `max_cos` 中位数 | `max_cos` 最大值 |
| --- | --- | --- | --- | --- | --- |
| 可答题 | 74 | 0.0170 | 0.0288 | 0.6403 | 0.8381 |
| **知识库外** | 10 | 0.0000 | **0.0064** | 0.4005 | **0.4449** |

两条信号在两组之间有清晰间隔：知识库外题的最大 `ts_rank` 0.0064 < 阈值 0.008，
最大余弦 0.4449 < 阈值 0.62；而可答题只要**任一条**越过阈值就不拒答
（可答题里 `max_cos` 最低只有 0.4953，靠 `ts_rank` 一路救回来）。

---

## 7. 系统指标

| 指标 | 值 | 说明 |
| --- | --- | --- |
| 端到端 P50 延迟 | **1048 ms** | 含 DeepSeek 网络往返，配置 D（84 条各跑 1 次） |
| 端到端 P95 延迟 | **2217 ms** | 只统计到生成阶段成功返回的请求 |
| 生成阶段 P50 / P95 | 971 ms / 2089 ms | 纯 API 往返 |
| 检索阶段 P50 / P95 | 120 ms / 177 ms | 含向量编码 + 两条召回 + 融合 + 重排 |
| 其中重排 P50 / P95 | 2 ms / 18 ms | 分数融合式重排（确定性） |
| 单问 prompt / completion tokens | 833.6 / 91.7 | 取自 API 返回的 `usage` |
| 74 题合计 tokens | prompt 70019 / completion 7702 | — |
| 单问成本 | ≈ **¥0.0006** | 按 2026-09-27 DeepSeek 公开定价换算（flash：输入 ¥0.5/百万、输出 ¥2/百万）；仅换算，未核对账单 |
| 索引构建耗时（冷缓存） | 约 **410 s**（2619 chunk，bge-m3，CPU，max_seq_length=512） | 含 embedding 编码 |
| 增量入库（语料无变化） | **0.1 s**（540 篇全部跳过、0 chunk 编码） | `content_hash` 判重 |
| embedding 缓存命中后的全量重建 | **6.8 s**（2619 chunk 全部用缓存向量） | 键 = embedding 模型 + 语料 content_hash 集合 |

> 缓存命中率未测：评测跑的是唯一问题，命中率完全由测试构造决定，写上去没有信息量。
> 上一版报告里的「增量入库（1 篇变化）14.4 s」本轮没有重测（重测需要改动库内某篇文档，
> 会连带改动 HNSW 图，使本页其余数字失效），因此**本页不再给出该数字**。

---

## 8. 门禁真实拦截记录（CI 里 1 次 + 本地降级 2 次）

门禁规则（`eval/baseline.json` + `src/eval/runner.py:check_gate`）：
`recall@5 / recall@10 / MRR@10 / 首条命中率` 允许 -2pt 抖动；`refusal_acc` **不得下降**；
`cite_hit` 允许 -2pt；`cite_halluc_rate` 不得 +2pt；`p95_latency_ms` / `avg_prompt_tokens` 不得恶化超过 20%。

### 8.1 CI 里的真实拦截（GitHub Actions，可点开复核）

| 项 | 值 |
| --- | --- |
| PR | [#1](https://github.com/x1247897956/sentinel-rag/pull/1) —— **负对照实验，刻意不合并** |
| 失败运行 | [run 36291977414](https://github.com/x1247897956/sentinel-rag/actions/runs/36291977414)（`pull_request`，结论 `failure`） |
| 改了什么 | 把 `hybrid_rerank` 的重排分支短路（`if mode == "hybrid_rerank" and False:`），其余不动 |
| 失败的步骤 | 第 15 步「生成指标（由冻结回答快照确定性重算）+ 门禁」 |
| 关键点 | 第 1~14 步（建库 → 入库 → 四组检索）全部 `success`，**红的只有门禁这一步**，所以不是环境/网络抖动 |

`gh run view 36291977414 --log-failed` 的原始输出（节选，逐字）：

```
[gate] 检索指标（hybrid_rerank）：{"n_answerable": 74, "recall@5": 0.8378, "recall@10": 0.9324, "mrr@10": 0.6487, "first_hit@1": 0.4595}
  hybrid         {"n_answerable": 74, "recall@5": 0.8378, "recall@10": 0.9324, "mrr@10": 0.6487, "first_hit@1": 0.4595}
  hybrid_rerank  {"n_answerable": 74, "recall@5": 0.8378, "recall@10": 0.9324, "mrr@10": 0.6487, "first_hit@1": 0.4595}
[gate] ❌ 未通过，掉线项：
  - recall@5: 0.8378 < baseline 0.8649 - 0.02
  - mrr@10: 0.6487 < baseline 0.7345 - 0.02
  - first_hit@1: 0.4595 < baseline 0.6081 - 0.02
##[error]Process completed with exit code 1.
```

`hybrid_rerank` 与 `hybrid` 两行完全相同，正是「重排确实被关掉了」的直接证据；
三条掉线项里 `mrr@10` 掉得最多（-8.6pt），与 §4.3 「重排的主战场是 MRR」一致。

同一次改动在本地（`--configs hybrid hybrid_rerank --judge`）的 Badcase 转移：
`context_truncated 5 → 6`、失败总数 `9 → 10`、`points_partial 0.7249 → 0.7136`、
`faithfulness 0.8939 → 0.8636`——关掉重排后有 1 条 gold 被挤出上下文预算。

### 8.2 本地降级复现（两条，一条命令可重跑）

`uv run python scripts/degrade_experiments.py --baseline eval/baseline.json` 的真实输出：

| 实验 | 改了什么 | 门禁结果 | 掉线项 | Badcase 转移 |
| --- | --- | --- | --- | --- |
| **E1 候选池 30 → 5** | `--topk-recall 5`（召回池缩小） | ❌ 退出码 1 | `recall@10: 0.8514 < baseline 0.9459 - 0.02` | `context_truncated 5 → 0`、`retrieval_miss 4 → 11`、`generation_halluc 0 → 10`，失败总数 9 → 21 |
| **E2 拒答阈值关掉** | `--refusal-min-fts 0 --refusal-min-cosine 0` | ❌ 退出码 1 | `refusal_acc: 0.0 < baseline 1.0` | 知识库外 10 条全部不再拒答 |

E1 的归因转移正是「候选池变小 ⇒ gold 更容易完全进不了池子」的预期结果；
E2 命中「拒答不得下降」这一条。两次的完整日志与退出码见 `eval/results/degrade_summary.json`。

> **口径说明**：8.1 是 **CI 里**的真实拦截（Actions 历史可查）；8.2 是**本地门禁脚本**
> 真实返回退出码 1，命令与输出都在上面、可复现。两者跑的是同一段 `check_gate` 逻辑。
>
> **另一个如实记录**：CI 与本地对同一份代码给出的 `hybrid` 指标并不逐位相同
> （CI `MRR@10` 0.6487 vs 本地 0.6464，差约 1 道题）。原因是 HNSW 是**近似**检索，
> 换机器/换插入顺序会让个别近似并列的候选换序。2pt 的门禁容差正是为这类抖动留的；
> 要做逐位复现需改用精确检索（见 §10-7）。

---

## 9. 复现

```bash
make db-up                 # PostgreSQL 16 + pgvector 0.8.6
uv run python scripts/prepare_corpus.py --verify-only   # 校验语料 sha256（应为 540 篇一致）
make setup                 # 装依赖（含本地 BGE 模型）
make ingest-all            # 建索引：2619 chunk 构建、2618 条入库（1 条同文档内重复文本被去重）
make eval                  # 四组对照 + 生成 + 判分（约 12 分钟）
make gate                  # 与 eval/baseline.json 比对，掉线退出码 1
uv run python scripts/degrade_experiments.py --baseline eval/baseline.json   # 复现两次降级拦截
```

**验证「同一条命令得到同一组数字」**（本轮修掉缓存量化缺陷后新增的自检）：

```bash
# 1) 缓存命中与未命中必须产出逐位相同的索引
rm -f data/cache/*.jsonl && uv run python -m src.ingest.ingest --all   # 冷缓存（约 410 s 编码）
docker exec sentinel-rag-db psql -U sentinel -d sentinel -t -A \
  -c "SELECT md5(string_agg(chunk_id || embedding::text, '|' ORDER BY chunk_id)) FROM chunks;"
uv run python -m src.ingest.ingest --all                               # 热缓存（约 7 s）
docker exec sentinel-rag-db psql -U sentinel -d sentinel -t -A \
  -c "SELECT md5(string_agg(chunk_id || embedding::text, '|' ORDER BY chunk_id)) FROM chunks;"   # 必须相同

# 2) 两次独立检索评测必须给出完全相同的四组指标
uv run python -m src.eval.runner --configs vector fts hybrid hybrid_rerank --no-generate --tag a
uv run python -m src.eval.runner --configs vector fts hybrid hybrid_rerank --no-generate --tag b
```

本轮的实测结果：两次索引哈希均为 `bd89bb7f58dd310be3f6913e21357e2b`；
两次检索评测四组指标完全一致（`vector` 0.3108/0.3378/0.2462/0.2027、
`fts` 0.8514/0.9324/0.7861/0.7297、`hybrid` 0.8378/0.9324/0.6464/0.4595、
`hybrid_rerank` 0.8649/0.9459/0.7345/0.6081）。

CI 侧（`.github/workflows/eval.yml`）用**冻结的回答快照** `eval/snapshot/answers.jsonl`
与 `eval/snapshot/judge.jsonl` 确定性重算生成指标，因此**不需要在 CI 里放 LLM 密钥**；
检索指标仍在 CI 里真跑（建库 → 入库 → 四组检索）。快照里每条都带
`answer / citations / allowed_ids / refused / gold_chunk_ids`，可逐条人工复核。

---

## 10. 已知偏差与限制

1. **`D > C > max(A,B)` 的形状没有完全成立**：见 §4.3。`D > C` 成立、`C > A` 成立，
   但 **B 的 `MRR@10` / 首条命中率高于 C 与 D**。根因是向量一路在本语料上太弱
   （精确标识符查询对稠密检索天然不友好），并已用「补编号只抬 ~0.07 余弦」的实测否掉了
   一个看似可行的改法。正确的修法是换更强的多语种/英文域 embedding，**本轮没做**。
2. **多跳题几乎没有召回到**（`recall@5` 0.00~0.25，n=4）：样本量小，但方向明确——
   需要查询分解或按编号做 metadata 过滤。**本轮没做**。
3. **拒答阈值在回归集上标定，没有独立 held-out 集**：存在过拟合风险；阈值一起写进了 `src/config.py`，
   改动会同时影响 `runs` 表里的历史记录（`runs` 表有 `config` 字段但没有单独存阈值）。
4. **人工抽检只有 14 条**：一致率 0.9286 的 95% 置信区间较宽（约 ±0.13），
   不足以支撑「判分模型可靠」这种强结论，只能说「在本轮抽检里 13/14 一致」。
   而且抽检是**本人**做的，不是独立第三方——这是自评，不是外部审计。
5. **生成指标由快照重算**：CI 里的生成指标来自本地一次真实运行冻结下来的回答，
   不是 CI 现场调用模型；这样做是为了不在 CI 放密钥，代价是 CI 不会发现「模型行为变了」。
6. **语料是快照**：CVE/公告会持续更新，本报告只对 `corpus_manifest.json` 记录的那份快照负责。
7. **HNSW 是近似检索**：向量一路的结果依赖索引图。本轮已把「缓存量化不一致」这个会让图变化的
   缺陷修掉，并验证了冷/热缓存重建逐位相同；但**换机器、换插入顺序或换 pgvector 版本仍可能改变
   个别近似并列候选的顺序**。这不是推测：同一份代码在 CI 上给出 `hybrid` 的 `MRR@10` 0.6487，
   在本地给出 0.6464（差约 1 道题，见 §8.1）。门禁的 2pt 容差正是为这类抖动留的。
   要做逐位复现需改用精确（暴力）检索，千级 chunk 下代价可接受——**本轮没换**。
8. **本报告不含任何旧项目（LLM Guard 等）的评测数字**，也不含估算值。
