"""人工抽检：从生成阶段的答案快照里按 20% 抽样，产出人工复核表，并计算一致率。

两种用法：
  1) 生成抽检表（含模型判分结论与理由）：
       uv run python scripts/human_audit.py sample --rate 0.2
  2) 人工把 `human_covered` 填好后（true / false），计算一致率：
       uv run python scripts/human_audit.py agree

抽检表落在 reports/human_audit.csv；填好后本脚本输出一致率，写进 docs/eval-report.md §6。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVAL_DIR, REPORT_DIR  # noqa: E402

SNAPSHOT = EVAL_DIR / "snapshot"
AUDIT = REPORT_DIR / "human_audit.csv"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"缺少 {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_sample(args) -> int:
    answers = {a["qid"]: a for a in load_jsonl(SNAPSHOT / "answers.jsonl")}
    judge = {j["qid"]: j for j in load_jsonl(SNAPSHOT / "judge.jsonl")}
    rows = []
    for qid, j in judge.items():
        if j.get("points_partial") is None:
            continue
        a = answers.get(qid, {})
        rows.append(
            {
                "qid": qid,
                "question": a.get("question", ""),
                "gold_points": " | ".join(a.get("gold_points", [])),
                "answer": (a.get("answer") or "").replace("\n", " "),
                "llm_points_partial": j.get("points_partial"),
                "llm_points_full": j.get("points_full"),
                "llm_faithful": j.get("faithful"),
                "llm_reason": (j.get("reason") or "").replace("\n", " "),
                "judge_model": j.get("judge_model"),
                "human_points_partial": "",   # ← 人工填写：覆盖要点比例 0~1
                "human_points_full": "",      # ← 人工填写：true / false
                "human_note": "",
            }
        )
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    n = max(1, int(round(len(rows) * args.rate)))
    sample = sorted(rows[:n], key=lambda r: r["qid"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sample)
    print(f"抽检 {len(sample)}/{len(rows)} 条（{args.rate:.0%}）→ {AUDIT}")
    print("请人工核对 `human_points_partial` / `human_points_full` 两列后运行：")
    print("  uv run python scripts/human_audit.py agree")
    return 0


def cmd_agree(args) -> int:
    if not AUDIT.exists():
        raise SystemExit(f"缺少 {AUDIT}，先运行 sample")
    with open(AUDIT, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("human_points_partial", "").strip() != ""]
    if not rows:
        raise SystemExit("抽检表还没填 human_* 列")
    agree = 0
    err = []
    for r in rows:
        try:
            human = float(r["human_points_partial"])
        except ValueError:
            continue
        auto = float(r["llm_points_partial"])
        # 一致判定：两边落在同一档（全对 / 部分对 / 没对）
        def bucket(x: float) -> str:
            return "full" if x >= 0.999 else ("none" if x <= 0.001 else "partial")

        ok = bucket(human) == bucket(auto)
        agree += ok
        if not ok:
            err.append((r["qid"], auto, human))
    total = len(rows)
    rate = agree / total if total else 0.0
    print(f"人工抽检 {total} 条，分档一致 {agree} 条，一致率 {rate:.4f}")
    for qid, auto, human in err:
        print(f"  不一致：{qid} 判分={auto} 人工={human}")
    out = {"n": total, "agree": agree, "agreement_rate": round(rate, 4), "disagreements": err}
    (REPORT_DIR / "human_audit_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--rate", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=20260926)
    sp.set_defaults(func=cmd_sample)
    sa = sub.add_parser("agree")
    sa.set_defaults(func=cmd_agree)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
