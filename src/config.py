"""统一配置入口：从环境变量 / .env 读取，提供默认值与可复现性元信息。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(ROOT)

load_dotenv(ROOT / ".env", override=False)

# 分块与检索参数：写进评测报告，改动即视为新配置
CHUNK_TARGET = int(os.getenv("CHUNK_TARGET", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "cross_encoder")
EMBED_DIM = 1024

RECALL_TOPK = int(os.getenv("RECALL_TOPK", "30"))
FINAL_TOPN = int(os.getenv("FINAL_TOPN", "6"))
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2600"))
# 拒答判据必须用**绝对可比**的量，不能用候选池内归一化后的相对分（池内归一的
# 冠军恒等于 1.0，阈值永远失效）。这里用两条绝对信号的组合：
#   max_fts   = 候选池里最大的 ts_rank（强标识符命中），无答案时接近 0
#   max_cosine= 候选池里最大的向量余弦相似度，多语种模型的相似度量级稳定
REFUSAL_MIN_FTS = float(os.getenv("REFUSAL_MIN_FTS", "0.008"))
REFUSAL_MIN_COSINE = float(os.getenv("REFUSAL_MIN_COSINE", "0.62"))
# 兼容旧变量名：若显式设置了 REFUSAL_MIN_SCORE，则只用它作为 fts 阈值
REFUSAL_MIN_SCORE = float(os.getenv("REFUSAL_MIN_SCORE", "0.008"))

GEN_MODEL = os.getenv("GEN_MODEL", "deepseek-chat")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "")

CORPUS_LIMITS = {
    "cve": int(os.getenv("CORPUS_CVE_LIMIT", "320")),
    "ghsa": int(os.getenv("CORPUS_GHSA_LIMIT", "120")),
    "attack": int(os.getenv("CORPUS_ATTACK_LIMIT", "60")),
    "owasp": int(os.getenv("CORPUS_OWASP_LIMIT", "40")),
}


@dataclass
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://sentinel:sentinel@localhost:55432/sentinel"
        )
    )
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    gen_model: str = GEN_MODEL
    judge_model: str = JUDGE_MODEL
    judge_base_url: str = JUDGE_BASE_URL

    def require_llm(self) -> tuple[str, str]:
        if not self.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置（复制 .env.example 为 .env）")
        return self.deepseek_base_url, self.deepseek_api_key


def get_settings() -> Settings:
    return Settings()


RAW_DIR = ROOT / "data" / "raw"
CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_DIR = ROOT / "eval"
REPORT_DIR = ROOT / "reports"
