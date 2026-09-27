#!/usr/bin/env bash
# 端到端冒烟：假设 db-up 与 ingest 已经跑过。
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
UV="${UV:-uv}"

echo "== 库内状态 =="
$UV run python -m src.cli stats

echo
echo "== 分词生效验证（整句当 token 命中 0 行 vs jieba 预分词） =="
$UV run python -m src.cli probe "身份认证绕过漏洞"

echo
echo "== 知识库内问题（配置 D） =="
$UV run python -m src.cli ask "CVE-2026-100599 影响哪些版本？" --mode hybrid_rerank

echo
echo "== 知识库外问题（应拒答） =="
$UV run python -m src.cli ask "宋代汝窑天青釉的烧制温度区间是多少？" --mode hybrid_rerank

echo
echo "== 四组检索对照（同一问题的候选来源差异） =="
for m in vector fts hybrid hybrid_rerank; do
  $UV run python -m src.cli search "MinIO 受影响版本区间是什么？" --mode "$m" | head -6
  echo "---"
done

echo
echo "== FastAPI 健康检查 =="
$UV run python - <<'PY'
from fastapi.testclient import TestClient
from src.service.app import app
c = TestClient(app)
r = c.get('/health')
print(r.status_code, r.json()['status'], r.json()['chunks'], 'chunks')
PY
