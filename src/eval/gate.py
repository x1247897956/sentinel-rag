"""CI 门禁：用「真实检索结果」+「冻结的模型回答快照」重算指标并与基线比较。

为什么要有这个脚本：
  - 检索指标（recall / MRR）在 CI 里是**真跑**的（建库 → 入库 → 四组检索）；
  - 生成指标需要 LLM API key，而 CI 里放长期密钥不安全，因此把带完整证据字段的
    回答快照（answer / citations / allowed_ids / refused / gold_chunk_ids）冻结进仓库，
    由本脚本**确定性重算**引用命中率、引用幻觉率、拒答正确率、过度拒答率；
  - 要点覆盖率取冻结的判分结论（judge.jsonl），并在报告中标注判分模型与抽检一致率。

掉线（recall@5 / MRR@10 / cite_hit / refusal_acc）或显著恶化（p95 延迟、prompt tokens）→ 退出码 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.runner import check_gate  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generation_metrics(answers: list[dict], judge: list[dict] | None = None) -> dict:
    judge_by_qid = {j["qid"]: j for j in (judge or [])}
    answerable = [a for a in answers if a.get("answerable", True)]
    unanswerable = [a for a in answers if not a.get("answerable", True)]
    answered = [a for a in answerable if not a.get("refused")]
    with_cites = [a for a in answered if a.get("citations")]
    cite_hit = (
        round(sum(1 for a in with_cites if set(a["citations"]) & set(a.get("gold_chunk_ids", []))) / len(with_cites), 4)
        if with_cites
        else None
    )
    halluc = sum(
        1
        for a in answerable
        if a.get("hallucinated_citations")
        or [c for c in a.get("citations", []) if c not in set(a.get("allowed_ids", []))]
    )
    out = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "cite_hit": cite_hit,
        "cite_halluc_rate": round(halluc / len(answerable), 4) if answerable else None,
        "no_citation_rate": round(sum(1 for a in answered if not a.get("citations")) / len(answerable), 4)
        if answerable
        else None,
        "refusal_acc": round(sum(1 for a in unanswerable if a.get("refused")) / len(unanswerable), 4)
        if unanswerable
        else None,
        "over_refusal": round(sum(1 for a in answerable if a.get("refused")) / len(answerable), 4)
        if answerable
        else None,
    }
    if judge_by_qid:
        partials = [j["points_partial"] for j in judge_by_qid.values() if j.get("points_partial") is not None]
        fulls = [j["points_full"] for j in judge_by_qid.values() if j.get("points_full") is not None]
        out["points_partial"] = round(sum(partials) / len(partials), 4) if partials else None
        out["points_full"] = round(sum(fulls) / len(fulls), 4) if fulls else None
        # 与 runner.aggregate_generation 口径一致：判分缺失（None）不计入分母
        faithful = [j["faithful"] for j in judge_by_qid.values() if j.get("faithful") is not None]
        out["faithfulness"] = round(sum(1 for f in faithful if f) / len(faithful), 4) if faithful else None
        out["judged_items"] = len(judge_by_qid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", type=Path, required=True, help="runner --no-generate 产出的 eval_*.json")
    ap.add_argument("--answers", type=Path, default=None, help="冻结的答案快照 jsonl")
    ap.add_argument("--judge", type=Path, default=None, help="冻结的判分结果 jsonl")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--main-config", default="hybrid_rerank")
    args = ap.parse_args()

    payload = json.loads(args.retrieval.read_text(encoding="utf-8"))
    results = payload["results"]
    main = results[args.main_config]
    metrics = {
        "retrieval": main["retrieval"],
        "retrieval_by_category": main.get("retrieval_by_category", {}),
        "system": main["system"],
    }
    print(f"[gate] 检索指标（{args.main_config}）：{json.dumps(metrics['retrieval'], ensure_ascii=False)}")
    print("[gate] 四组对照：")
    for cfg, r in results.items():
        print(f"  {cfg:14s} {json.dumps(r['retrieval'], ensure_ascii=False)}")

    if args.answers:
        if not args.answers.is_file():
            print(f"[gate] ❌ 回答快照不存在：{args.answers}")
            return 2
        answers = load_jsonl(args.answers)
        dataset_path = Path(__file__).resolve().parents[2] / "eval/dataset/regression_set.jsonl"
        dataset = load_jsonl(dataset_path)
        expected = {item["qid"]: item.get("gold_chunk_ids", []) for item in dataset}
        actual = {item["qid"]: item.get("gold_chunk_ids", []) for item in answers}
        if len(actual) != len(answers) or actual != expected:
            print("[gate] ❌ 回答快照的 qid/gold 标注与当前回归集不完全一致")
            return 2
        judge = None
        if args.judge:
            if not args.judge.is_file():
                print(f"[gate] ❌ 判分快照不存在：{args.judge}")
                return 2
            judge = load_jsonl(args.judge)
            judge_qids = [item.get("qid") for item in judge]
            if len(set(judge_qids)) != len(judge_qids) or not set(judge_qids) <= set(actual):
                print("[gate] ❌ 判分快照包含重复或不在回答快照中的 qid")
                return 2
        metrics["generation"] = generation_metrics(answers, judge)
        print(f"[gate] 生成指标（来自冻结快照）：{json.dumps(metrics['generation'], ensure_ascii=False)}")
    else:
        print("[gate] ❌ 必须提供本次回答快照；禁止复制基线生成指标充当本次指标")
        return 2

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_metrics = baseline.get("results", {}).get(baseline.get("main_config", args.main_config), baseline)
    ok, violations = check_gate(metrics, base_metrics, tolerance=baseline.get("tolerance", 0.02))
    if ok:
        print("[gate] ✅ 通过：所有门禁指标不低于基线")
        return 0
    print("[gate] ❌ 未通过，掉线项：")
    for v in violations:
        print(f"  - {v}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
