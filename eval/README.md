# eval/ 目录说明

| 路径 | 内容 | 是否进 git |
| --- | --- | --- |
| `dataset/regression_set.jsonl` | **回归集 84 条**（单跳 55 / 概念 15 / 多跳 4 / 知识库外 10），带 `gold_chunk_ids` 与人工核对过的 `gold_points` | ✅ |
| `dataset/_draft/` | 起草过程的中间产物（含被丢弃的候选与丢弃原因） | ✅（留痕） |
| `baseline.json` | **门禁基线**：上次被认可的指标快照 + 容差（2pt） | ✅ |
| `results/eval_v4.json` | **当前有效**的一次完整评测结果（四组对照 + 生成指标 + 归因） | ✅ |
| `results/rows_v4.jsonl` | 当前评测的逐题明细（检索轨迹、引用、判分、归因）；`.gitignore` 默认排除 `rows_*.jsonl`，这一份用 `git add -f` 显式纳入 | ✅ |
| `results/badcase_v4.md` | 自动生成的 Badcase 四分类报表 | ✅ |
| `results/eval_v3.json` / `badcase_v3.md` | **已被 v4 取代**的上一版结果，保留用于留痕 | ✅ |
| `results/rows_v3.jsonl` | 同上；体积原因未纳入 git（可用 `make eval` 重跑） | ❌ |
| `results/eval_degrade_*.json` | 两次降级复现实验的检索结果 | ✅ |
| `results/degrade_summary.json` | 降级实验汇总（含门禁退出码与 Badcase 转移） | ✅ |
| `snapshot/answers.jsonl` | **冻结的模型回答**（含 `citations` / `allowed_ids` / `refused` / `gold_chunk_ids`），CI 用它确定性重算生成指标 | ✅ |
| `snapshot/judge.jsonl` | 冻结的判分结论（要点覆盖、faithfulness、判分理由） | ✅ |
| `snapshot/corpus_manifest.json` | 语料快照清单（篇数 + 整文件 sha256） | ✅ |
| `results/traces_*.json` | 检索轨迹缓存（两阶段评测用，体积大、可重跑） | ❌ 本地生成 |

## 为什么有 v3 和 v4 两份结果

重跑评测时发现了一个**会让同一条命令产出不同数字**的缺陷：embedding 缓存把向量
`round(x, 6)` 落盘，而入库时 `_vec_literal` 按 `%.7f` 格式化——于是「缓存命中」与
「缓存未命中」写进库的向量差在第 6/7 位小数，近似检索（HNSW）因此把个别近似并列的
候选排出不同顺序（实测纯向量 `recall@5` 0.3108 vs 0.2973）。

- **v3**：修之前的数字，其中一部分不可复现（`hybrid` 的 MRR 等），已作废；
- **v4**：修之后的数字，**`docs/eval-report.md` 与 `README.md` 使用的就是这一版**。

修法与验证（冷/热缓存重建逐位相同、两次独立评测指标完全一致）见
`docs/eval-report.md` 开头的「复现性声明」与 §9。保留 v3 是为了留下这次修正的痕迹，
不是为了留一个可以挑着用的数字池——**任何对外引用都只用 v4**。

## 数字的唯一出处

`docs/eval-report.md` 里出现的每个指标，都能在 `results/eval_v4.json`（数字）与
`results/rows_v4.jsonl`（逐题证据）里找到对应项；语料规模在 `snapshot/corpus_manifest.json`
与 `wc -l data/corpus/*.jsonl` 双重可查。

## 复现

```bash
make db-up && make fetch && make ingest-all  # 重建同一份索引
uv run python scripts/prepare_corpus.py --verify-only   # 校验语料快照
make eval                                   # 跑四组对照 + 生成 + 判分
make gate                                   # 与 baseline.json 比对
uv run python scripts/degrade_experiments.py --baseline eval/baseline.json
```
