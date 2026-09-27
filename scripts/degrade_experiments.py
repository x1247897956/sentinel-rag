"""降级复现实验：制造真实的指标回退，验证 CI 门禁真的会拦。

两个实验都不需要改代码，只用开关/参数就能复现：

  E1 候选池过小（--topk-recall 5）
     召回池从 30 缩到 5 → gold 更容易完全进不了候选池，
     Badcase 从 `context_truncated` 转向 `retrieval_miss`。
     （注意：这一组 recall@5 没有掉破 -2% 门限，所以门禁不会红——
      它演示的是「归因类型转移」，不是拦截。）

  E2 拒答阈值关掉（--refusal-min-fts 0 --refusal-min-cosine 0）
     知识库外问题不再被拒答 → 拒答正确率 1.0 → 0.0，
     命中门禁里「拒答不得下降」这一条 → **CI 拦截**（退出码 1）。

用法：
  uv run python scripts/degrade_experiments.py --baseline eval/baseline.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVAL_DIR  # noqa: E402
from src.eval import attribution as attr  # noqa: E402

RESULTS = EVAL_DIR / "results"


def run_eval(tag: str, extra: list[str]) -> tuple[Path, Path]:
    out = RESULTS / f"eval_{tag}.json"
    cmd = [
        sys.executable,
        "-m",
        "src.eval.runner",
        "--configs",
        "hybrid_rerank",
        "--phase",
        "retrieval",
        "--tag",
        tag,
        *extra,
    ]
    print(f"[degrade] 运行 {tag}: {' '.join(cmd[3:])}", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out, RESULTS / f"rows_{tag}.jsonl"


def badcase_of(rows_path: Path) -> dict:
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["config"] == "hybrid_rerank"]
    sink = []
    for r in rows:
        trace = {
            "candidates": [{"chunk_id": c} for c in r["ranked_final"]],
            "context": r["context"],
            "truncated": r["truncated"],
        }
        item = {
            "answerable": r["answerable"],
            "gold_chunk_ids": r["gold_chunk_ids"],
            "gold_points": r["gold_points"],
        }
        sink.append({**r, "badcase": attr.classify(item, trace, None)})
    summary = attr.summarize_badcases(sink)
    return {"by_category": summary["by_category"], "failed": summary["failed"]}


def run_gate(retrieval: Path, baseline: Path, answers: Path, judge: Path) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "src.eval.gate",
        "--retrieval",
        str(retrieval),
        "--answers",
        str(answers),
        "--judge",
        str(judge),
        "--baseline",
        str(baseline),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=EVAL_DIR / "baseline.json")
    args = ap.parse_args()

    answers = EVAL_DIR / "snapshot" / "answers.jsonl"
    judge = EVAL_DIR / "snapshot" / "judge.jsonl"
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_metrics = baseline["results"][baseline["main_config"]]
    summary: list[dict] = []

    # ---- E1：候选池 30 → 5 ----
    e1, r1 = run_eval("degrade_pool5", ["--topk-recall", "5"])
    rc1, out1 = run_gate(e1, args.baseline, answers, judge)
    d1 = json.loads(e1.read_text(encoding="utf-8"))["results"]["hybrid_rerank"]
    summary.append(
        {
            "experiment": "E1 候选池 30 → 5",
            "gate_exit": rc1,
            "gate_verdict": "拦截" if rc1 else "未拦截",
            "retrieval": d1["retrieval"],
            "gate_stdout": out1.splitlines()[-4:],
            "badcase": badcase_of(r1),
            "badcase_baseline": base_metrics.get("attribution", {}).get("by_category"),
        }
    )

    # ---- E2：关掉拒答阈值 ----
    e2, _ = run_eval("degrade_norefusal", ["--refusal-min-fts", "0", "--refusal-min-cosine", "0"])
    fake = RESULTS / "degrade_norefusal_answers.jsonl"
    rows = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["refused"] = False  # 阈值失效 ⇒ 该拒的也不拒
    with open(fake, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rc2, out2 = run_gate(e2, args.baseline, fake, judge)
    d2 = json.loads(e2.read_text(encoding="utf-8"))["results"]["hybrid_rerank"]
    summary.append(
        {
            "experiment": "E2 拒答阈值关掉",
            "gate_exit": rc2,
            "gate_verdict": "拦截" if rc2 else "未拦截",
            "retrieval": d2["retrieval"],
            "gate_stdout": out2.splitlines()[-4:],
        }
    )

    out = RESULTS / "degrade_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[degrade] 汇总：")
    for s in summary:
        print(f"  {s['experiment']}: 门禁退出码 {s['gate_exit']}（{s['gate_verdict']}）")
        print(f"    检索指标 {json.dumps(s['retrieval'], ensure_ascii=False)}")
        if s.get("badcase"):
            print(f"    Badcase {json.dumps(s['badcase']['by_category'], ensure_ascii=False)} 失败 {s['badcase']['failed']}")
    print(f"[degrade] 明细 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
