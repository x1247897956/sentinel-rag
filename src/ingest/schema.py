"""语料文档的统一表示。

每个 Doc 对应「一篇公开安全语料」；`to_record()` 产出落盘 JSONL 的形状。
`content_hash` 是增量更新与去重的判据：内容没变 → 跳过；内容变了 → 新记录 + 旧版本 stale。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Doc:
    doc_id: str
    source: str
    source_type: str
    title: str
    text: str                      # 规范化后的正文（用于分块与去重）
    published_at: str | None = None
    updated_at: str | None = None
    cve_id: str | None = None
    severity: str | None = None
    cvss: float | None = None
    affected_versions: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        rec = asdict(self)
        rec["content_hash"] = sha256_text(self.text.strip())
        return rec

    def to_json(self) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False, sort_keys=True)


def write_jsonl(path, docs: list[Doc]) -> tuple[int, str]:
    """写入 JSONL 并返回 (条数, 整文件 sha256)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [d.to_json() for d in docs]
    body = "\n".join(lines) + "\n"
    path.write_text(body, encoding="utf-8")
    return len(lines), hashlib.sha256(body.encode("utf-8")).hexdigest()
