"""分块：按语料结构分类型，而不是按固定字符数无脑切。

四条硬约束（设计文档 §2.3，也是 Badcase 归因的前提）：
  1) CVE/GHSA 以语义段（描述 / 受影响版本区间 / CVSS / 修复建议）为天然边界；
  2) **版本区间不得切散**——切散了模型会答错影响面；
  3) 每个 chunk 都带 doc_id / source / section 等元数据，否则归因做不了；
  4) chunk_id = f"{doc_id}#{idx}"，是引用溯源的最小单位。

打包顺序：section 边界 → 行/句原子边界 → 只在单行仍超长时才用字符窗口兜底。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import jieba

from src.config import CHUNK_OVERLAP, CHUNK_TARGET
from src.ingest.schema import Doc, sha256_text

jieba.setLogLevel(20)

# 目标长度按「估算 token」计（中文 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token）
TARGET_TOKEN_ENV = CHUNK_TARGET


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    idx: int
    text: str
    tokens: str
    section: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def finalize(self) -> "Chunk":
        self.content_hash = sha256_text(self.text.strip())
        return self


def tokenize(text: str) -> str:
    """应用层 jieba 预分词 → 空格连接的 token 串，供 to_tsvector('simple', tokens) 使用。

    为什么不装 zhparser：需要自编译镜像，本地与容器编译产物不一致；放应用层后
    检索链路与部署都变成确定的（代价是多存一列）。
    """
    return " ".join(t for t in jieba.lcut_for_search(text) if t.strip())


def _estimate_tokens(text: str) -> int:
    """粗略 token 预算：中文按字、英文按 4 字符 1 token 估算，仅用于上下文预算与分块。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def _hard_window(text: str, target: int, overlap: int) -> list[str]:
    """兜底：单行/单段本身就超长时，按字符窗口切（尽量在标点处断开）。"""
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + target)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("；"), window.rfind("\n"), window.rfind(". "))
            if cut > target // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _atomic_units(body: str, target: int, overlap: int) -> list[str]:
    """把 section 正文拆成「不跨越语义边界」的原子单元：行 → 句 → 超长兜底窗口。"""
    units: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _estimate_tokens(line) <= target:
            units.append(line)
            continue
        # 长行按句号/分号切，保留标点
        sentences = re.split(r"(?<=[。！？；;.!?])\s*", line)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if _estimate_tokens(s) <= target:
                units.append(s)
            else:
                units.extend(_hard_window(s, target, overlap))
    return units


def _pack(section: str, body: str, target: int, overlap: int) -> list[tuple[str, str]]:
    """在一个 section 内按目标长度打包原子单元，不跨 section。"""
    units = _atomic_units(body, target, overlap)
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_len = 0
    for u in units:
        ulen = _estimate_tokens(u)
        if buf and buf_len + ulen > target:
            out.append((section, "\n".join(buf)))
            # 重叠：把上一个 chunk 的尾部（约 overlap 个 token）带进下一个
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(buf):
                tail.insert(0, prev)
                tail_len += _estimate_tokens(prev)
                if tail_len >= overlap:
                    break
            buf = tail if tail_len < target else []
            buf_len = sum(_estimate_tokens(x) for x in buf)
        buf.append(u)
        buf_len += ulen
    if buf:
        out.append((section, "\n".join(buf)))
    return out


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """按 `## 小标题` 切分保留语义边界；无小标题时整体作为一节。"""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_title = "正文"
    buf: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if buf and "\n".join(buf).strip():
                sections.append((current_title, "\n".join(buf)))
            current_title = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if buf and "\n".join(buf).strip():
        sections.append((current_title, "\n".join(buf)))
    if not sections:
        return [("正文", text)]
    return sections


def chunk_document(doc: Doc, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """按语料结构分类型分块（当前四类语料都带 `## ` 结构标记，走同一套 section 打包）。"""
    sections = _split_markdown_sections(doc.text)
    packed: list[tuple[str, str]] = []
    for section, body in sections:
        packed.extend(_pack(section, body, target, overlap))

    chunks: list[Chunk] = []
    for idx, (section, body) in enumerate(packed):
        text = body.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{idx}",
                doc_id=doc.doc_id,
                idx=idx,
                text=text,
                tokens=tokenize(f"{doc.title}\n{text}"),
                section=section,
            ).finalize()
        )
    if not chunks:
        text = doc.text.strip()
        chunks = [
            Chunk(
                chunk_id=f"{doc.doc_id}#0",
                doc_id=doc.doc_id,
                idx=0,
                text=text,
                tokens=tokenize(f"{doc.title}\n{text}"),
                section="全文",
            ).finalize()
        ]
    return chunks


def chunk_corpus(docs: list[Doc], target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, target=target, overlap=overlap))
    return chunks
