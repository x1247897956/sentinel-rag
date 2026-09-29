# SentinelRAG 评测报告

本报告只记录本轮命令实际产生的结果。检索与拒答指标由脚本确定性计算；未运行的语义判分与人工复核明确标为未测。

## 1. 指标定义

设可答问题数为 $N$，问题 $i$ 的人工/规则标注 gold chunk 集合为 $G_i$，有序检索结果为 $R_i$。

| 指标 | 定义 |
| --- | --- |
| `recall@k` | $\frac{1}{N}\sum_i \mathbb{1}[G_i \cap R_i^{(k)} \ne \varnothing]$ |
| `MRR@10` | $\frac{1}{N}\sum_i 1/\mathrm{rank}_i$；前十名无 gold 时记 0 |
| 首条命中率 | $\frac{1}{N}\sum_i \mathbb{1}[\mathrm{rank}_i=1]$ |
| 引用命中率 | 有引用且未拒答的可答回答中，引用至少包含一个 gold chunk 的比例 |
| 引用幻觉率 | 可答题中，输出过上下文以外 chunk 引用的回答比例 |
| 拒答正确率 | 知识库外题中最终拒答的比例 |
| 过度拒答率 | 可答题中最终拒答的比例 |
| 要点字面匹配下界 | 非拒答回答里，gold point 被代码子串规则命中的要点数 / 要点总数；不代表语义覆盖 |

## 2. 环境与数据

| 项 | 本轮实测 |
| --- | --- |
| 环境 | Apple Silicon macOS，CPU 推理，Python 3.11，PostgreSQL 16 + pgvector 0.8.6 |
| embedding | `BAAI/bge-m3`，本地快照 revision `5617a9f61b028005a4858fdac845db406aefb181` |
| 重排 | `BAAI/bge-reranker-base`，交叉编码器，本地快照 revision `2cfc18c9415c912f9d8155881c133215df768a70` |
| 生成请求模型 | 请求 `deepseek-chat`；85 次实际 API 请求返回 `deepseek-flash`。其余 15 题在生成前被阈值拒答 |
| 判分模型 | 未用于本轮。仓库旧配置指向 DeepSeek 模型；它与生成模型属于同一服务商，不能称为不同源 |
| Prompt | `answer-zh-v1.0` |
| 回归集 | `eval/dataset/regression_set.jsonl`，100 行；SHA-256 `9f6a6bec0e6b6576d746bec55e7c4ee4aa3873e7ab6c47c0a90878811e011544` |
| 回答快照 | `eval/snapshot/answers.jsonl`；本轮 100 行。SHA-256 以仓库命令复核 |
| 上下文 | 召回池 30，最终最多 6 段，预算 2600 tokens；分块目标 420、重叠 80 |

### 数据集和来源

| 来源 | JSONL 行数 / 文档数 |
| --- | ---: |
| CVE List v5 | 320 |
| GitHub Advisory Database（OSV） | 120 |
| MITRE ATT&CK STIX | 60 |
| OWASP Cheat Sheet Series | 40 |
| 合计 | 540 |

回归集共 100 题：单跳事实 55、概念解释 25、多跳 8、知识库外 12；可答题 88。题目与 `gold_points` 的构造及逐字校验规则见 `scripts/build_regression.py`。逐题独立人工核验本轮未完成；知识库外题由项目作者选择，尚无独立 held-out 集。

## 3. 四组检索对照

同一份索引、同一回归集的 88 道可答题。四项检索指标均由代码确定性计算，不经 LLM。

| 配置 | recall@5 | recall@10 | MRR@10 | 首条命中率 |
| --- | ---: | ---: | ---: | ---: |
| A 纯向量 | 0.3750 | 0.4205 | 0.3075 | 0.2614 |
| B 纯全文 | 0.8295 | 0.8977 | 0.7661 | 0.7159 |
| C 混合 RRF | 0.8636 | 0.9432 | 0.6885 | 0.5227 |
| D 混合 + cross-encoder | 0.5000 | 0.6477 | 0.4418 | 0.3750 |

本轮没有达到 `D > C > max(A,B)`：交叉编码器把召回与排序都拉低。D 相对 C 的 MRR@10 下降，检索 P50 为 1070 ms、其中重排 P50 为 813 ms。当前实验不支持“交叉编码器改善结果”的结论；模型和语料语言不匹配是可能原因，尚未单独验证。

## 4. 生成结果与边界

| 指标 | 值 | 说明 |
| --- | ---: | --- |
| 引用命中率 | 0.9286 | 有效引用回答 42 条中 39 条命中 gold |
| 引用幻觉率 | 0.0000 | 输出引用都经过实际上下文校验；上下文外引用会被拒绝 |
| 知识库外拒答正确率 | 1.0000 | 12 条均拒答 |
| 可答题过度拒答率 | 0.5227 | 88 条中 46 条最终拒答；阈值拒答 3 条，另有无有效上下文引用的回答被拒绝 |
| 要点字面匹配下界 | 0.4655 | 只在 42 条非拒答回答上按子串规则计算，不是语义覆盖率 |
| 要点全字面匹配率 | 0.1905 | 同上，不是语义全覆盖率 |
| 语义要点覆盖 / faithfulness | 未测 | 本轮未运行判分模型 |
| 本轮人工复核 | 未完成 | 需要人工阅读答案与对应来源片段 |

同一服务商的另一个模型不能替代独立来源的 judge；仓库里较早版本的人工审核表也不应用来代表本轮答案。当前代码会把没有引用或引用不属于送入模型的上下文的输出替换为拒答，保证返回答案的引用有效，但这明显推高了过度拒答率。

系统观测：端到端 P50/P95 为 1850/3486 ms；检索阶段（含 embedding 与重排）P50/P95 为 1070/1944 ms；重排 P50/P95 为 813/1361 ms。85 次 API 调用平均 prompt/completion 为 891.9/86.1 tokens。数值来自本轮运行结果文件。

## 5. Badcase 归因

本轮 D 配置的自动分类器记录 42 个失败：`retrieval_miss` 4、`rerank_misorder` 0、`context_truncated` 38、`generation_halluc` 0。大量 gold chunk 虽在候选池里，却被交叉编码器排序或上下文预算挤出；分类器没有把引用校验失败单独列类，因此生成阶段的过度拒答另在上节报告。

## 6. 复现

数据与当前运行的实际命令：

```bash
wc -l data/corpus/*.jsonl eval/dataset/regression_set.jsonl
shasum -a 256 eval/dataset/regression_set.jsonl
make db-up
make ingest
UV_CACHE_DIR=.uv-cache HF_HOME=.hf-cache uv run python -m src.eval.runner --configs vector fts hybrid hybrid_rerank --rerank-backend cross_encoder --tag cross_encoder_verified --snapshot-dir eval/snapshot_cross_encoder
```

本轮完整评测开始时使用隔离目录保留既有快照；在核对文件与报告后，将新回答快照复制到 `eval/snapshot/answers.jsonl` 供 CI 使用。评测 JSON 保存在 `eval/results/eval_cross_encoder_verified.json`。CI 重新运行检索，并从当前回归集匹配回答快照中的 qid 和 gold 标注；缺快照或缺指标会失败。

## 7. CI 与已知限制

GitHub Actions 曾有一次真实负对照失败：run `36306724693`，PR 将重排短路后，检索门禁返回失败退出码。该历史运行使用当时的分数融合配置，不作为本轮交叉编码器的评测数据。本轮会以本次测量更新基线。

限制：本轮没有独立来源的 judge、没有新一轮人工复核、没有 held-out 数据；拒答阈值也在同一回归集上使用。D 配置未改善预期检索指标，且引用校验带来较高过度拒答率。这些结果应当保留为当前实现的失败证据。
