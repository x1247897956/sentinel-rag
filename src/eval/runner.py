"""评测 runner：一条命令跑完四组对照 + 检索/生成指标 + Badcase 归因 + 门禁判定。

用法：
  # 只跑检索对照（确定性、不调 LLM，CI 友好）
  uv run python -m src.eval.runner --configs vector fts hybrid hybrid_rerank --no-generate

  # 完整评测（含生成指标与 LLM 判分）
  uv run python -m src.eval.runner --configs vector fts hybrid hybrid_rerank \\
      --generate-config hybrid_rerank --judge --baseline eval/baseline.json

门禁：读 eval/baseline.json，任一指标掉线即退出码 1（CI 就是靠这个 fail 的）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    CONTEXT_TOKEN_BUDGET,
    EMBED_MODEL,
    EVAL_DIR,
    FINAL_TOPN,
    RECALL_TOPK,
    REFUSAL_MIN_COSINE,
    REFUSAL_MIN_FTS,
    RERANK_MODEL,
    get_settings,
)
from src.eval import attribution as attr
from src.eval import metrics as M
from src.eval import prompts
from src.eval.regression import point_in_text, read_jsonl
from src.retrieval import storage
from src.retrieval.hybrid import (
    RRF_WEIGHTS,
    Retriever,
    citation_hallucination,
    extract_citations,
)
from src.retrieval.models import Embedder, LLMClient, Reranker

DATASET = EVAL_DIR / "dataset" / "regression_set.jsonl"
BASELINE = EVAL_DIR / "baseline.json"
RESULTS = EVAL_DIR / "results"


def dataset_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_config(
    retriever: Retriever,
    items: list[dict],
    config: str,
    topk_recall: int,
    final_topn: int,
    token_budget: int,
    refusal_min_fts: float,
    refusal_min_cosine: float,
    rerank_backend: str,
    gen_config: str = "hybrid_rerank",
    generate: bool = False,
    judge: bool = False,
    llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    answer_sink: list[dict] | None = None,
    judge_sink: list[dict] | None = None,
    trace_cache: dict[str, Any] | None = None,
    full_trace_sink: dict[str, Any] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        if trace_cache is not None and item["qid"] in trace_cache and config == gen_config:
            trace = trace_cache[item["qid"]]
        else:
            trace = retriever.retrieve(
            q,
            mode=config,
            topk_recall=topk_recall,
            topn_final=final_topn,
            token_budget=token_budget,
            refusal_min_fts=refusal_min_fts,
            refusal_min_cosine=refusal_min_cosine,
            rerank_backend=rerank_backend,
            llm=llm,
            )
        trace_dict = trace.to_dict()
        if full_trace_sink is not None:
            full_trace_sink[item["qid"]] = {
                **trace_dict,
                "candidates": [c.to_dict(full=True) for c in trace.candidates],
            }
        row: dict[str, Any] = {
            "qid": item["qid"],
            "question": q,
            "category": item["category"],
            "answerable": item.get("answerable", True),
            "gold_chunk_ids": item.get("gold_chunk_ids", []),
            "gold_points": item.get("gold_points", []),
            # D（含重排）用重排后的顺序算检索指标：重排本来就是为了改顺序，
            # 用重排前的顺序算就永远看不到重排的价值。同时保留重排前顺序供对照。
            "retrieved": trace.recall_order if config != gen_config else trace.retrieved,
            "retrieved_prererank": trace.recall_order,
            "ranked_final": trace.retrieved,
            "context": trace_dict["context"],
            "truncated": trace_dict["truncated"],
            "top_score": trace.top_score,
            "refused": trace.refused,
            "retrieval_ms": trace.total_ms,
            "rerank_ms": trace.rerank_ms,
            "config": config,
        }
        row["recall@5"] = M.recall_at_k(row["retrieved"], row["gold_chunk_ids"], 5)
        row["recall@10"] = M.recall_at_k(row["retrieved"], row["gold_chunk_ids"], 10)
        row["mrr@10"] = M.reciprocal_rank(row["retrieved"], row["gold_chunk_ids"], 10)
        row["first_hit@1"] = 1.0 if M.first_hit_rank(row["retrieved"], row["gold_chunk_ids"], 10) == 1 else 0.0

        if generate and config == gen_config:
            try:
                gen = generate_answer(retriever, item, trace, llm, judge, judge_llm)
            except Exception as exc:  # noqa: BLE001
                gen = {
                    "answer": f"[生成失败] {exc}",
                    "citations": [],
                    "refused": False,
                    "cite_hit": None,
                    "cite_halluc": False,
                    "points_partial": None,
                    "points_full": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "gen_latency_ms": 0,
                    "latency_ms": trace.total_ms,
                    "prompt_version": prompts.PROMPT_VERSION,
                    "gen_error": str(exc)[:200],
                }
                print(f"  [!] {item['qid']} 生成失败：{str(exc)[:120]}", flush=True)
            row.update(gen)
            if answer_sink is not None:
                answer_sink.append(
                    {
                        "qid": item["qid"],
                        "question": q,
                        "category": item["category"],
                        "answerable": item.get("answerable", True),
                        "gold_chunk_ids": item.get("gold_chunk_ids", []),
                        "gold_points": item.get("gold_points", []),
                        "answer": gen["answer"],
                        "citations": gen["citations"],
                        "allowed_ids": gen.get("allowed_ids", []),
                        "refused": gen["refused"],
                        "top_score": trace.top_score,
                        "top_chunk": trace.context[0].chunk_id if trace.context else None,
                        "gen_model": llm.model if llm else None,
                        "prompt_version": prompts.PROMPT_VERSION,
                    }
                )
            if judge_sink is not None and item.get("gold_points") and not gen.get("refused"):
                judge_sink.append(
                    {
                        "qid": item["qid"],
                        "points_partial": gen.get("points_partial"),
                        "points_full": gen.get("points_full"),
                        "faithful": gen.get("judge_faithful"),
                        "reason": gen.get("judge_reason", ""),
                        "method": gen.get("points_method", "llm_judge"),
                        "judge_model": (judge_llm or llm).model if (judge_llm or llm) else None,
                    }
                )
        row["badcase"] = attr.classify(item, trace_dict, row if generate else None)
        rows.append(row)
        if i % 10 == 0:
            print(f"  [{config}] {i}/{len(items)}", flush=True)
    return rows


def generate_answer(
    retriever: Retriever,
    item: dict,
    trace,
    llm: LLMClient | None,
    judge: bool,
    judge_llm: LLMClient | None,
) -> dict[str, Any]:
    assert llm is not None
    if trace.refused:
        return {
            "answer": prompts.REFUSAL_TEXT,
            "citations": [],
            "refused": True,
            "cite_hit": None,
            "cite_halluc": False,
            "points_partial": None,
            "points_full": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "gen_latency_ms": 0,
            "latency_ms": trace.total_ms,
            "prompt_version": prompts.PROMPT_VERSION,
        }
    blocks = [f"[{c.chunk_id}] 来源={c.meta.get('source_type')} 文档={c.meta.get('doc_id')}\n{c.text}" for c in trace.context]
    msgs = prompts.build_answer_messages(item["question"], blocks)
    t0 = time.time()
    resp = llm.chat(msgs, temperature=0.0, max_tokens=700)
    gen_ms = int((time.time() - t0) * 1000)
    answer = (resp["choices"][0]["message"]["content"] or "").strip()
    citations = extract_citations(answer)
    allowed = [c.chunk_id for c in trace.context]
    halluc = citation_hallucination(citations, trace.retrieved)
    gold = set(item.get("gold_chunk_ids", []))
    cite_hit = None
    if citations:
        cite_hit = 1.0 if (set(citations) & gold) else 0.0
    out: dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "refused": False,
        "cite_hit": cite_hit,
        "cite_halluc": bool(halluc),
        "hallucinated_citations": halluc,
        "allowed_ids": allowed,
        "points_partial": None,
        "points_full": None,
        "prompt_tokens": resp.get("_usage", {}).get("prompt_tokens", 0),
        "completion_tokens": resp.get("_usage", {}).get("completion_tokens", 0),
        "gen_latency_ms": gen_ms,
        "latency_ms": trace.total_ms + gen_ms,
        "prompt_version": prompts.PROMPT_VERSION,
    }
    if judge and judge_llm is not None and item.get("gold_points"):
        try:
            verdict = judge_llm.json(
                prompts.build_judge_messages(item["gold_points"], answer), temperature=0.0, max_tokens=2500
            )
            covered = [bool(x) for x in verdict.get("covered", [])]
            if len(covered) < len(item["gold_points"]):
                covered += [False] * (len(item["gold_points"]) - len(covered))
            n = len(item["gold_points"])
            out["points_partial"] = round(sum(1 for c in covered[:n] if c) / n, 4)
            out["points_full"] = 1.0 if all(covered[:n]) else 0.0
            out["judge_reason"] = str(verdict.get("reason", ""))[:300]
            out["judge_faithful"] = bool(verdict.get("faithful", True))
        except Exception as exc:  # noqa: BLE001
            out["judge_error"] = str(exc)[:200]
    # 确定性兜底：判分不可用时用「要点逐字/子串命中」做下界估计
    if out["points_partial"] is None and item.get("gold_points"):
        hit = sum(1 for p in item["gold_points"] if any(point_in_text(p, s) for s in [answer]))
        out["points_partial"] = round(hit / len(item["gold_points"]), 4)
        out["points_full"] = 1.0 if hit == len(item["gold_points"]) else 0.0
        out["points_method"] = "substring"
    return out


def aggregate_generation(rows: list[dict]) -> dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r.get("refused")]
    with_cites = [r for r in answered if r.get("citations")]
    out: dict[str, Any] = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_refused_on_answerable": sum(1 for r in answerable if r.get("refused")),
        "cite_hit": round(sum(r["cite_hit"] for r in with_cites if r.get("cite_hit") is not None) / len(with_cites), 4)
        if with_cites
        else None,
        "cite_halluc_rate": round(sum(1 for r in answered if r.get("cite_halluc")) / len(answerable), 4)
        if answerable
        else None,
        "refusal_acc": round(sum(1 for r in unanswerable if r.get("refused")) / len(unanswerable), 4)
        if unanswerable
        else None,
        "over_refusal": round(sum(1 for r in answerable if r.get("refused")) / len(answerable), 4)
        if answerable
        else None,
        "no_citation_rate": round(sum(1 for r in answered if not r.get("citations")) / len(answerable), 4)
        if answerable
        else None,
    }
    partial = [r["points_partial"] for r in answered if r.get("points_partial") is not None]
    full = [r["points_full"] for r in answered if r.get("points_full") is not None]
    out["points_partial"] = round(sum(partial) / len(partial), 4) if partial else None
    out["points_full"] = round(sum(full) / len(full), 4) if full else None
    faithful = [r["judge_faithful"] for r in answered if "judge_faithful" in r]
    out["faithfulness"] = round(sum(1 for x in faithful if x) / len(faithful), 4) if faithful else None
    return out


def check_gate(metrics: dict, baseline: dict, tolerance: float = 0.02) -> tuple[bool, list[str]]:
    """门禁：召回/引用/拒答不得掉线；延迟与 token 不得恶化超过 20%。"""
    violations: list[str] = []
    ret = metrics.get("retrieval", {})
    gen = metrics.get("generation", {})
    sysm = metrics.get("system", {})
    b_ret = baseline.get("retrieval", {})
    b_gen = baseline.get("generation", {})
    b_sys = baseline.get("system", {})

    for key in ("recall@5", "mrr@10", "recall@10", "first_hit@1"):
        if key in b_ret and key in ret and ret[key] < b_ret[key] - tolerance:
            violations.append(f"{key}: {ret[key]} < baseline {b_ret[key]} - {tolerance}")
    if b_gen.get("refusal_acc") is not None and gen.get("refusal_acc") is not None:
        if gen["refusal_acc"] < b_gen["refusal_acc"]:
            violations.append(f"refusal_acc: {gen['refusal_acc']} < baseline {b_gen['refusal_acc']}")
    if b_gen.get("cite_hit") is not None and gen.get("cite_hit") is not None:
        if gen["cite_hit"] < b_gen["cite_hit"] - tolerance:
            violations.append(f"cite_hit: {gen['cite_hit']} < baseline {b_gen['cite_hit']} - {tolerance}")
    if b_gen.get("cite_halluc_rate") is not None and gen.get("cite_halluc_rate") is not None:
        if gen["cite_halluc_rate"] > b_gen["cite_halluc_rate"] + 0.02:
            violations.append(
                f"cite_halluc_rate: {gen['cite_halluc_rate']} > baseline {b_gen['cite_halluc_rate']} + 0.02"
            )
    for key, factor in (("p95_latency_ms", 1.2), ("avg_prompt_tokens", 1.2)):
        if key in b_sys and key in sysm and b_sys[key] and sysm[key] > b_sys[key] * factor:
            violations.append(f"{key}: {sysm[key]} > baseline {b_sys[key]} × {factor}")
    return (not violations), violations


def _trace_from_dict(td: dict):
    """把缓存里的轨迹字典还原成轻量 Trace（generation 阶段复用，避免重复检索）。"""
    from src.retrieval.hybrid import Candidate, Trace

    cands = []
    for c in td.get("candidates", []):
        cands.append(
            Candidate(
                chunk_id=c["chunk_id"],
                text=c.get("text", ""),
                rrf_score=c.get("rrf_score", 0.0),
                vector_score=c.get("vector_score"),
                fts_score=c.get("fts_score"),
                rerank_score=c.get("rerank_score"),
                rank_vector=c.get("rank_vector"),
                rank_fts=c.get("rank_fts"),
                final_rank=c.get("final_rank"),
                truncated=bool(c.get("truncated")),
                meta={"source_type": c.get("source"), "doc_id": c.get("doc_id"), "section": c.get("section")},
            )
        )
    context_ids = set(td.get("context", []))
    return Trace(
        mode=td.get("mode", "hybrid_rerank"),
        candidates=cands,
        context=[c for c in cands if c.chunk_id in context_ids],
        recall_ms=td.get("recall_ms", 0),
        rerank_ms=td.get("rerank_ms", 0),
        total_ms=td.get("total_ms", 0),
        top_score=td.get("top_score", 0.0),
        refused=bool(td.get("refused")),
        recall_ranking=td.get("recall_ranking", []),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["vector", "fts", "hybrid", "hybrid_rerank"])
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--generate-config", default="hybrid_rerank")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--topk-recall", type=int, default=RECALL_TOPK)
    ap.add_argument("--topn-final", type=int, default=FINAL_TOPN)
    ap.add_argument("--token-budget", type=int, default=CONTEXT_TOKEN_BUDGET)
    ap.add_argument("--refusal-min-fts", type=float, default=REFUSAL_MIN_FTS)
    ap.add_argument("--refusal-min-cosine", type=float, default=REFUSAL_MIN_COSINE)
    ap.add_argument("--rerank-backend", default="score_fusion", choices=["score_fusion", "cross_encoder", "llm"])
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None, help="结果文件名后缀，用于标记不同实验")
    ap.add_argument("--phase", default="all", choices=["all", "retrieval", "generation"],
                    help="retrieval=只跑四种配置的检索指标并缓存轨迹；generation=复用轨迹只跑生成+判分")
    args = ap.parse_args()

    items = read_jsonl(args.dataset)
    if args.limit:
        items = items[: args.limit]
    ds_sha = dataset_sha(args.dataset)
    print(f"[eval] 数据集 {args.dataset}（{len(items)} 条，sha256={ds_sha[:16]}…）")

    conn = storage.connect()
    retriever = Retriever(conn, embedder=Embedder(), reranker=Reranker())
    settings = get_settings()
    llm = None
    judge_llm = None
    if not args.no_generate or args.rerank_backend == "llm":
        llm = LLMClient(model=settings.gen_model)
    if args.judge:
        if settings.judge_model and settings.judge_model != settings.gen_model:
            judge_llm = LLMClient(model=settings.judge_model, base_url=settings.judge_base_url or None)
        else:
            judge_llm = llm
            print("[eval] 注意：未配置不同源的判分模型，判分与生成同源（报告中已标注）")

    tag0 = args.tag or f"{int(time.time())}"
    trace_cache: dict[str, Any] = {}
    trace_path = RESULTS / f"traces_{tag0}.json"
    if args.phase == "generation" and trace_path.exists():
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
        for qid, td in raw.items():
            trace_cache[qid] = _trace_from_dict(td)
        print(f"[eval] 复用 {len(trace_cache)} 条检索轨迹 → {trace_path}")

    results: dict[str, Any] = {}
    rows_by_config: dict[str, list[dict]] = {}
    answer_sink: list[dict] = []
    judge_sink: list[dict] = []
    full_trace_sink: dict[str, Any] = {}
    if args.phase == "retrieval":
        args.no_generate = True
    for cfg in args.configs:
        print(f"[eval] 配置 {cfg} 开始", flush=True)
        generate = (not args.no_generate) and cfg == args.generate_config
        rows = run_config(
            retriever,
            items,
            cfg,
            args.topk_recall,
            args.topn_final,
            args.token_budget,
            args.refusal_min_fts,
            args.refusal_min_cosine,
            args.rerank_backend,
            gen_config=args.generate_config,
            generate=generate,
            judge=args.judge,
            llm=llm,
            judge_llm=judge_llm,
            answer_sink=answer_sink if generate else None,
            judge_sink=judge_sink if generate else None,
            trace_cache=trace_cache if generate else None,
            full_trace_sink=full_trace_sink if (not generate and cfg == args.generate_config) else None,
        )
        rows_by_config[cfg] = rows
        # 检索指标用「全部候选池」算（recall@k 只到前 k，候选池 >= 10 即可）
        results[cfg] = {
            "retrieval": M.aggregate_retrieval(rows),
            "retrieval_by_category": M.aggregate_retrieval_by_category(rows),
            "system": M.aggregate_system(rows),
        }
        if generate:
            results[cfg]["generation"] = aggregate_generation(rows)
            results[cfg]["attribution"] = attr.summarize_badcases(rows)
        print(f"[eval] {cfg}: {json.dumps(results[cfg]['retrieval'], ensure_ascii=False)}")

    if args.phase in {"retrieval", "all"} and args.generate_config in rows_by_config:
        RESULTS.mkdir(parents=True, exist_ok=True)
        cache = dict(full_trace_sink)
        trace_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] 检索轨迹缓存 → {trace_path}（{len(cache)} 条，供 generation 阶段复用）")

    # 主配置（D）的指标即门禁比较对象
    main_cfg = args.generate_config if not args.no_generate else args.configs[-1]
    main_metrics = results[main_cfg]
    payload = {
        "eval_id": str(uuid.uuid4()),
        "configs": args.configs,
        "main_config": main_cfg,
        "dataset": str(args.dataset),
        "dataset_sha256": ds_sha,
        "n_items": len(items),
        "git_sha": git_sha(),
        "env": {
            "embed_model": EMBED_MODEL,
            "rerank": {"score_fusion": f"score_fusion(weights={RRF_WEIGHTS})",
                        "cross_encoder": RERANK_MODEL,
                        "llm": f"llm:{settings.gen_model}"}[args.rerank_backend],
            "gen_model": settings.gen_model,
            "judge_model": settings.judge_model or f"{settings.gen_model}(同源)",
            "chunk_target": CHUNK_TARGET,
            "chunk_overlap": CHUNK_OVERLAP,
            "topk_recall": args.topk_recall,
            "topn_final": args.topn_final,
            "token_budget": args.token_budget,
            "refusal_min_fts": args.refusal_min_fts,
            "refusal_min_cosine": args.refusal_min_cosine,
            "prompt_version": prompts.PROMPT_VERSION,
        },
        "results": results,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{int(time.time())}"
    out_path = args.out or (RESULTS / f"eval_{tag}.json")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(RESULTS / f"rows_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for cfg, rows in rows_by_config.items():
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval] 结果落盘 → {out_path}")

    # 冻结快照：CI 用它在无密钥的前提下确定性重算生成指标
    snap_dir = EVAL_DIR / "snapshot"
    if answer_sink:
        snap_dir.mkdir(parents=True, exist_ok=True)
        with open(snap_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
            for a in answer_sink:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"[eval] 回答快照 → {snap_dir / 'answers.jsonl'}（{len(answer_sink)} 条）")
    if judge_sink:
        with open(snap_dir / "judge.jsonl", "w", encoding="utf-8") as fh:
            for j in judge_sink:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"[eval] 判分快照 → {snap_dir / 'judge.jsonl'}（{len(judge_sink)} 条）")

    with conn.cursor() as cur:
        storage.log_eval_run(
            cur,
            {
                "eval_id": payload["eval_id"],
                "config": main_cfg,
                "dataset_sha": ds_sha,
                "metrics": main_metrics,
                "attribution": main_metrics.get("attribution", {}),
                "git_sha": payload["git_sha"],
            },
        )
        conn.commit()
    conn.close()

    # 门禁
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        base_metrics = baseline.get("results", {}).get(baseline.get("main_config", main_cfg), baseline)
        ok, violations = check_gate(main_metrics, base_metrics, tolerance=baseline.get("tolerance", 0.02))
        print("[gate] " + ("通过" if ok else "未通过"))
        for v in violations:
            print(f"[gate] 掉线：{v}")
        if not ok:
            return 1
    else:
        print("[gate] 未提供 --baseline，跳过门禁比较")

    if main_metrics.get("attribution"):
        md = attr.markdown_report(main_metrics["attribution"])
        (RESULTS / f"badcase_{tag}.md").write_text(md, encoding="utf-8")
        print(f"[eval] Badcase 报表 → {RESULTS / f'badcase_{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
