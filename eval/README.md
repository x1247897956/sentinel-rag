# eval/ 目录说明

| 路径 | 内容 | 是否进 git |
| --- | --- | --- |
| `dataset/regression_set.jsonl` | **回归集 84 条**（单跳 55 / 概念 15 / 多跳 4 / 知识库外 10），带 `gold_chunk_ids` 与人工核对过的 `gold_points` | ✅ |
| `dataset/_draft/` | 起草过程的中间产物（含被丢弃的候选与丢弃原因） | ✅（留痕） |
| `baseline.json` | **门禁基线**：上次被认可的指标快照 + 容差（2pt） | ✅ |
| `results/eval_v3.json` | 最终一次完整评测的结果（四组对照 + 生成指标 + 归因） | ✅ |
| `results/rows_v3.jsonl` | 最终评测的逐题明细（检索轨迹、引用、判分） | ✅ |
| `results/badcase_v3.md` | 自动生成的 Badcase 四分类报表 | ✅ |
| `results/eval_degrade_*.json` | 两次降级复现实验的检索结果 | ✅ |
| `results/degrade_summary.json` | 降级实验汇总（含门禁退出码与 Badcase 转移） | ✅ |
| `snapshot/answers.jsonl` | **冻结的模型回答**（含 `citations` / `allowed_ids` / `refused` / `gold_chunk_ids`），CI 用它确定性重算生成指标 | ✅ |
| `snapshot/judge.jsonl` | 冻结的判分结论（要点覆盖、faithfulness、判分理由） | ✅ |
| `snapshot/corpus_manifest.json` | 语料快照清单（篇数 + 整文件 sha256） | ✅ |
| `results/traces_*.json` | 检索轨迹缓存（两阶段评测用，体积大、可重跑） | ❌ 本地生成 |

## 数字的唯一出处

`docs/eval-report.md` 里出现的每个指标，都能在 `results/eval_v3.json`（数字）与
`results/rows_v3.jsonl`（逐题证据）里找到对应项；语料规模在 `snapshot/corpus_manifest.json`
与 `wc -l data/corpus/*.jsonl` 双重可查。

## 复现

```bash
make db-up && make fetch && make ingest     # 重建同一份索引
uv run python scripts/prepare_corpus.py --verify-only   # 校验语料快照
make eval                                   # 跑四组对照 + 生成 + 判分
make gate                                   # 与 baseline.json 比对
uv run python scripts/degrade_experiments.py --baseline eval/baseline.json
```
