# SentinelRAG —— 安全知识库检索 Agent

| 仓库 | 方向 |
| --- | --- |
| **sentinel-rag（本仓库）** | 检索正确性 |
| [silver-guard](https://github.com/x1247897956/silver-guard) | 决策与对抗 |
| [dsh-llm-guard](https://github.com/x1247897956/dsh-llm-guard) | 输入输出护栏 |

SentinelRAG 将公开漏洞、安全公告和防护指南整理为可增量更新的知识库，通过 jieba 中文分词、PostgreSQL 全文检索与 pgvector 向量检索召回，再用 RRF 融合及可切换的重排器生成带 chunk 引用的回答。引用不在实际上下文或回答没有有效引用时，服务会拒答。

当前边界是单机、单租户和约两千六百个 chunk；没有线上部署或真实用户。实验没有得到预期的“重排改善排序”结果，细节与复现数据见[评测报告](docs/eval-report.md)。

## 当前实测

当前快照包含 540 篇公开语料，来源为 CVE List v5、GitHub Advisory Database、MITRE ATT&CK STIX 和 OWASP Cheat Sheet Series；100 题回归集有 88 道可答题和 12 道知识库外题。各文件行数、哈希和完整命令见评测报告。

| 配置 | recall@5 | recall@10 | MRR@10 | 首条命中率 |
| --- | ---: | ---: | ---: | ---: |
| A 纯向量 | 0.3750 | 0.4205 | 0.3075 | 0.2614 |
| B 纯全文 | 0.8295 | 0.8977 | 0.7661 | 0.7159 |
| C 混合 RRF | 0.8636 | 0.9432 | 0.6885 | 0.5227 |
| D 混合 + `bge-reranker-base` | 0.5000 | 0.6477 | 0.4418 | 0.3750 |

这次交叉编码器重排降低了检索指标。生成评测中，引用命中率为 0.9286，引用幻觉率为 0，知识库外拒答正确率为 1.0000，可答题过度拒答率为 0.5227。拒答策略会将无引用或上下文外引用的回答替换为拒答，这也造成较高的过度拒答率。要点只按字面匹配计算下界；语义覆盖率与忠实度未测，当前没有本轮答案的人工复核结果。

## 快速开始

```bash
make db-up
cp .env.example .env  # 设置 DEEPSEEK_API_KEY
make setup
make fetch
make ingest
make eval
make gate
```

启动服务：`make serve`，提供 `POST /ask`、`POST /ingest` 和 `GET /health`。

## 架构

```text
公开语料 → 解析与结构化分块 → content_hash 去重 / stale 失效 → PostgreSQL + pgvector
问题 → jieba → 全文召回 ─┐
问题 → BGE-m3 → 向量召回 ─┴→ RRF → BGE cross-encoder → 上下文预算
                                                        ↓
                              DeepSeek 回答 → 上下文引用校验 / 不满足则拒答
```

详见[架构说明](docs/architecture.md)与[设计记录](docs/design-notes.md)。

## 设计取舍与限制

- 用应用层 jieba 预分词后写入 `tokens` 与 `tsvector`，让 PostgreSQL 的全文检索能处理中文。
- embedding 使用 `BAAI/bge-m3`；D 组重排实测使用 `BAAI/bge-reranker-base`。在当前中英混合语料和中文问题上，交叉编码器排序变差且延迟上升；分数融合后端也保留为可对照选项。
- 检索指标由固定回归集确定性计算；生成 API 与检索分开测。由于判分和生成配置同属 DeepSeek，本轮没有启用 LLM 判分，也没有把旧版本的人工抽检结果用于本轮结论。
- 回归集的要点按原文子串验证，但逐题独立人工复核未完成；知识库外题由项目作者选择，尚无 held-out 集。
- 尚未加入工具扩展、长轮次对话与文档注入防护。

## 数据与许可

仅包含公开语料；语料数据仍遵循各来源自己的许可与条款。代码使用 MIT 许可。详见[来源及许可说明](docs/eval-report.md#数据集和来源)。
