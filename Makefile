.PHONY: help setup db-up db-down fetch ingest ingest-all eval eval-fast gate serve smoke test lint fmt clean reset

UV ?= uv
export UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.uv-python
export HF_HOME ?= $(CURDIR)/.hf-cache

help:
	@echo "SentinelRAG —— 常用命令"
	@echo "  make setup      建 venv 并装依赖（含本地 BGE embedding / 重排）"
	@echo "  make db-up      起 PostgreSQL + pgvector 容器并建表"
	@echo "  make fetch      抓取公开安全语料 → data/corpus/*.jsonl"
	@echo "  make ingest     增量入库（jieba 预分词 + BGE 编码 + 去重/stale）"
	@echo "  make eval       四组对照 + 检索/生成指标 + Badcase 归因"
	@echo "  make gate       用 eval/baseline.json 做门禁判定（掉线退出码 1）"
	@echo "  make serve      起 FastAPI（/ask /ingest /health）"
	@echo "  make test       跑单元测试"

setup:
	$(UV) venv --python 3.11
	$(UV) sync --extra local-models --extra dev

db-up:
	docker compose up -d
	@echo "等待 Postgres 就绪…"
	@for i in $$(seq 1 30); do docker exec sentinel-rag-db pg_isready -U sentinel -d sentinel >/dev/null 2>&1 && break || sleep 2; done
	docker exec sentinel-rag-db pg_isready -U sentinel -d sentinel

db-down:
	docker compose down

fetch:
	$(UV) run python -m src.ingest.fetch_corpus

ingest:
	$(UV) run python -m src.ingest.ingest

ingest-all:
	$(UV) run python -m src.ingest.ingest --all

eval:
	$(UV) run python -m src.eval.runner \
		--configs vector fts hybrid hybrid_rerank \
		--judge --tag local

eval-fast:
	$(UV) run python -m src.eval.runner \
		--configs vector fts hybrid hybrid_rerank --no-generate --tag fast

gate:
	$(UV) run python -m src.eval.runner --configs hybrid_rerank --baseline eval/baseline.json --tag gate

serve:
	$(UV) run uvicorn src.service.app:app --host 127.0.0.1 --port 8000

smoke:
	bash scripts/smoke.sh

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests scripts

fmt:
	$(UV) run ruff format src tests scripts

clean:
	rm -rf data/corpus data/chunks.jsonl reports eval/results .pytest_cache

reset:
	docker compose down -v
