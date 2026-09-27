# eval/ 目录说明

| 路径 | 内容 | 是否进 git |
| --- | --- | --- |
| `dataset/regression_set.jsonl` | **回归集 100 条**（单跳 55 / 概念 25 / 多跳 8 / 知识库外 12），带 `gold_chunk_ids` 与人工核对过的 `gold_points` | ✅ |
| `dataset/_draft/` | 起草过程的中间产物（含被丢弃的候选、丢弃原因，以及**扩增前的旧回归集逐字节备份**） | ✅（留痕） |
| `baseline.json` | **门禁基线**：上次被认可的指标快照 + 容差（2pt） | ✅ |
| `results/eval_v5.json` | **当前有效**的一次完整评测结果（四组对照 + 生成指标 + 归因） | ✅ |
| `results/rows_v5.jsonl` | 当前评测的逐题明细（检索轨迹、引用、判分、归因）；`.gitignore` 默认排除 `rows_*.jsonl`，这一份用 `git add -f` 显式纳入 | ✅ |
| `results/badcase_v5.md` | 自动生成的 Badcase 四分类报表 | ✅ |
| `results/eval_v4.json` / `badcase_v4.md` | **已被 v5 取代**的上一版结果，保留用于留痕 | ✅ |
| `results/rows_v4.jsonl` | 同上；已纳入 git（便于对照两版逐题差异） | ✅ |
| `results/eval_degrade_*.json` | 两次降级复现实验的检索结果 | ✅ |
| `results/degrade_summary.json` | 降级实验汇总（含门禁退出码与 Badcase 转移） | ✅ |
| `snapshot/answers.jsonl` | **冻结的模型回答**（含 `citations` / `allowed_ids` / `refused` / `gold_chunk_ids`），CI 用它确定性重算生成指标 | ✅ |
| `snapshot/judge.jsonl` | 冻结的判分结论（要点覆盖、faithfulness、判分理由） | ✅ |
| `snapshot/corpus_manifest.json` | 语料快照清单（篇数 + 整文件 sha256） | ✅ |
| `results/traces_*.json` | 检索轨迹缓存（两阶段评测用，体积大、可重跑） | ❌ 本地生成 |

## 为什么有 v3 / v4 / v5 三份结果

| 版本 | 回归集 | 为什么被取代 |
| --- | --- | --- |
| **v3** | 84 条 | 数字**不可复现**：embedding 缓存把向量 `round(x,6)` 落盘，入库却按 `%.7f` 格式化，两条路径写进库的向量不同位，近似检索因此换序（纯向量 `recall@5` 0.3108 vs 0.2973） |
| **v4** | 84 条 | 复现性已修，但**题型配比失衡**：多跳题只有 4 条，`sample_multi_hop` 有个 bug（每个产品只出一题），看不到混合检索在多跳上的收益 |
| **v5** | **100 条** | **当前有效**。只**新增**、不改旧题；修掉多跳抽样 bug 后多跳题达到语料上限 8 条 |

关键点：**v4 → v5 是把题目配比补全，不是"调数字"**。原 84 条**逐字节未改**，
备份在 `_draft/regression_set.*.bak.jsonl`；新增的 16 条走的是同一套
「`gold_points` 必须原文逐字可查」的纪律，知识库外的新题还必须**实测**过不了拒答阈值。
v4 的结果仍在 git 里（`eval_v4.json` / `rows_v4.jsonl`），可以逐题对照两版差异。

## 数字的唯一出处

`docs/eval-report.md` 里出现的每个指标，都能在 `results/eval_v5.json`（数字）与
`results/rows_v5.jsonl`（逐题证据）里找到对应项；语料规模在 `snapshot/corpus_manifest.json`
与 `wc -l data/corpus/*.jsonl` 双重可查。

## 复现

```bash
make db-up && make fetch && make ingest-all  # 重建同一份索引
uv run python scripts/prepare_corpus.py --verify-only   # 校验语料快照（应为 540 篇）
make eval                                   # 跑四组对照 + 生成 + 判分
make gate                                   # 与 baseline.json 比对（只读，不覆盖快照）
uv run python scripts/degrade_experiments.py --baseline eval/baseline.json
uv run python scripts/extend_regression.py --dry-run    # 复现"只新增不改旧题"的扩增流程
```
