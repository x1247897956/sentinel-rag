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
        n = len([line for line in body.decode("utf-8").splitlines() if line.strip()])
        if sha != meta["sha256"]:
            problems.append(f"{p.name} sha256 不一致：{sha[:16]}… != {meta['sha256'][:16]}…")
        if n != meta["documents"]:
            problems.append(f"{p.name} 篇数不一致：{n} != {meta['documents']}")
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
