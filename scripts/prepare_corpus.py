"""语料快照准备与校验。

`eval/snapshot/corpus_manifest.json` 冻结了本次评测所用语料的**篇数与整文件 sha256**。
CI / 复现者用它判断手上的 data/corpus 是否与评测报告一致：

  uv run python scripts/prepare_corpus.py --verify-only   # 只校验，不通过则退出码 1
  uv run python scripts/prepare_corpus.py                 # 校验 + 打印抓取指引
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CORPUS_DIR, EVAL_DIR  # noqa: E402

SNAPSHOT = EVAL_DIR / "snapshot" / "corpus_manifest.json"


def count_documents(path: Path) -> int:
    """按「一条 Doc 记录一行」的权威口径计数：以 `\\n` 切行，逐行 JSON 解析。

    ⚠️ 不能用 `str.splitlines()`：它除了 `\\n` 还会在 U+2028 / U+2029 等字符处断行，
    而 OWASP 正文里确实含有这两个字符（`owasp:DOM_based_XSS_Prevention_Cheat_Sheet`），
    于是一行会被数成三行 —— 这个 bug 曾让快照清单把 40 篇 OWASP 记成 42 篇、
    总数记成 542（README 与评测报告写的是 540），校验却「通过」。
    现在计数与 `load_corpus()` 的读取口径完全一致，并顺带校验每行可解析。
    """
    n = 0
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        json.loads(line)  # 解析失败即抛错：宁可红，也不要静默接受坏语料
        n += 1
    return n


def verify(manifest_path: Path = SNAPSHOT) -> tuple[bool, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for key, meta in manifest["sources"].items():
        p = CORPUS_DIR / f"{key}.jsonl"
        if not p.exists():
            problems.append(f"缺少 {p.name}（应为 {meta['documents']} 篇）")
            continue
        body = p.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        n = count_documents(p)
        if sha != meta["sha256"]:
            problems.append(f"{p.name} sha256 不一致：{sha[:16]}… != {meta['sha256'][:16]}…")
        if n != meta["documents"]:
            problems.append(f"{p.name} 篇数不一致：{n} != {meta['documents']}")
    total = sum(count_documents(CORPUS_DIR / f"{k}.jsonl") for k in manifest["sources"] if (CORPUS_DIR / f"{k}.jsonl").exists())
    if total != manifest.get("total_documents"):
        problems.append(f"总篇数不一致：实际 {total} != 清单 {manifest.get('total_documents')}")
    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--manifest", type=Path, default=SNAPSHOT)
    args = ap.parse_args()
    if not args.manifest.exists():
        print(f"[corpus] 未找到快照清单 {args.manifest}")
        return 1
    ok, problems = verify(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if ok:
        print(f"[corpus] 校验通过：{manifest['total_documents']} 篇，与快照一致")
        for k, v in manifest["sources"].items():
            print(f"  - {k}: {v['documents']} 篇 sha256={v['sha256'][:16]}…")
        return 0
    print("[corpus] 校验未通过：")
    for p in problems:
        print(f"  ! {p}")
    if not args.verify_only:
        print("\n请执行以下命令重新抓取语料：")
        print("  uv run python -m src.ingest.fetch_corpus")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
