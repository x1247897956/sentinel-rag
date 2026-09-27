"""语料快照清单的计数必须与语料文件本身一致。

这个测试是为一个真实的坑加的：`str.splitlines()` 除了 `\\n` 还会在 U+2028 / U+2029
等字符处断行，而 OWASP 正文里确实含这两个字符，于是
`eval/snapshot/corpus_manifest.json` 把 40 篇 OWASP 记成 42 篇、总数记成 542，
而 README / `docs/eval-report.md` 写的是 540 —— 文档与「校验通过」的清单互相矛盾。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_corpus import SNAPSHOT, count_documents  # noqa: E402
from src.config import CORPUS_DIR  # noqa: E402


def test_manifest_counts_match_corpus_files() -> None:
    manifest = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for key, meta in manifest["sources"].items():
        path = CORPUS_DIR / f"{key}.jsonl"
        assert path.exists(), f"缺少语料文件 {path.name}"
        assert count_documents(path) == meta["documents"], (
            f"{path.name} 实际 {count_documents(path)} 篇，清单写 {meta['documents']} 篇"
        )


def test_manifest_total_is_the_sum_of_sources() -> None:
    manifest = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert manifest["total_documents"] == sum(
        meta["documents"] for meta in manifest["sources"].values()
    )


def test_splitlines_would_have_been_wrong_on_owasp() -> None:
    """钉住这个坑：OWASP 语料确实含有会被 splitlines() 当作换行的字符。"""
    body = (CORPUS_DIR / "owasp.jsonl").read_text(encoding="utf-8")
    authoritative = count_documents(CORPUS_DIR / "owasp.jsonl")
    naive = len([line for line in body.splitlines() if line.strip()])
    assert authoritative == 40
    assert naive > authoritative, "若这里不再成立，说明语料已变，本节注释需同步更新"
