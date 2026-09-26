"""模型层：本地 BGE embedding / cross-encoder 重排，以及 DeepSeek（OpenAI 兼容）对话接口。

诚实原则：这里**只有一个真实实现路径**，不做静默降级。
- embedding：BAAI/bge-small-zh-v1.5（512 维，本地 CPU）
- 重排：BAAI/bge-reranker-base（cross-encoder，本地 CPU）；另有 LLM 重排作为可选项，
  评测报告里必须写明实际用的是哪一种。
- 生成：DeepSeek（OpenAI 兼容 /chat/completions）
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import httpx

from src.config import EMBED_DIM, EMBED_MODEL, RERANK_MODEL, get_settings


def _hf_cache_dir() -> str | None:
    """允许把 HF 缓存放进工作区（沙箱环境 ~/.cache 可能不可写）。"""
    return os.getenv("HF_HOME")


class Embedder:
    """BGE-m3（多语种，1024 维），输出已归一化向量（配合 pgvector 余弦距离）。

    `max_seq_length` 取 512：分块目标长度是 ~420 token，实测 512 已覆盖几乎所有 chunk，
    而编码吞吐比默认 8192 高一个量级（32 个真实 chunk：0.5s vs 3.9s）。
    """

    _lock = threading.Lock()

    def __init__(self, model_name: str = EMBED_MODEL, dim: int = EMBED_DIM, max_seq_length: int = 512) -> None:
        self.model_name = model_name
        self.dim = dim
        self.max_seq_length = max_seq_length
        self._model = None

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name, cache_folder=_hf_cache_dir(), device="cpu"
                    )
                    self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> list[list[float]]:
        model = self._ensure()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [v.tolist() for v in vecs]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class Reranker:
    """bge-reranker-base cross-encoder：对 (query, passage) 逐对打分。

    `max_length=256`：候选片段大多数在 256 token 以内，实测把逐对打分的延迟从
    ~9s/30 候选降到 ~2s/30 候选（CPU，Apple M 系列），代价是最长的那几个 chunk 被截断。
    """

    _lock = threading.Lock()

    def __init__(self, model_name: str = RERANK_MODEL, max_length: int = 256) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        self.model_name,
                        cache_folder=_hf_cache_dir(),
                        device="cpu",
                        max_length=self.max_length,
                    )
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._ensure()
        pairs = [(query, p) for p in passages]
        raw = model.predict(pairs, batch_size=16)
        return [float(x) for x in raw]


class LLMClient:
    """DeepSeek OpenAI 兼容接口。失败时抛异常，不返回编造的答案。"""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        s = get_settings()
        self.model = model or s.gen_model
        self.base_url = (base_url or s.deepseek_base_url).rstrip("/")
        self.api_key = api_key or s.deepseek_api_key
        self._usage: list[dict[str, int]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        retries: int = 3,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                t0 = time.time()
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                data["_latency_ms"] = int((time.time() - t0) * 1000)
                usage = data.get("usage") or {}
                data["_usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                self._usage.append(data["_usage"])
                return data
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败（重试 {retries} 次）：{last_err}")

    def text(self, messages: list[dict[str, str]], **kw) -> str:
        data = self.chat(messages, **kw)
        return (data["choices"][0]["message"]["content"] or "").strip()

    def json(self, messages: list[dict[str, str]], **kw) -> Any:
        data = self.chat(messages, **kw)
        content = (data["choices"][0]["message"]["content"] or "").strip()
        content = _strip_code_fence(content)
        return json.loads(content)

    @property
    def usage_totals(self) -> dict[str, int]:
        return {
            "prompt_tokens": sum(u["prompt_tokens"] for u in self._usage),
            "completion_tokens": sum(u["completion_tokens"] for u in self._usage),
            "calls": len(self._usage),
        }


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
