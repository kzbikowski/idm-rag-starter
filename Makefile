# Cienkie wrappery na `uv run rag ...` – na Windowsie bez make wpisz komendę z prawej strony.
# Dodatkowe argumenty: make ingest ARGS="--chunking fixed --recreate"

.PHONY: help setup pull up down check ingest ask agent eval test lint reset-cache

DATA ?=
Q ?=
ARGS ?=

help: ## lista komend
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup: ## instaluje zależności (uv) i tworzy .env
	uv sync
	@test -f .env || (cp .env.example .env && echo "Utworzono .env z .env.example")

pull: ## pobiera modele Ollamy z .env
	./scripts/pull_models.sh

up: ## uruchamia Qdranta (Docker)
	docker compose up -d
	@echo "Dashboard Qdranta: http://localhost:6333/dashboard"

down: ## zatrzymuje Qdranta (dane zostają w wolumenie)
	docker compose down

check: ## smoke test: Ollama, modele, Qdrant, embedding, RAG, tool calling
	uv run rag check

ingest: ## indeksuje dokumenty z DATA_DIR (.env) albo DATA=ścieżka
	uv run rag ingest $(DATA) $(ARGS)

ask: ## pytanie RAG: make ask Q="..."
	@test -n "$(Q)" || (echo 'Użycie: make ask Q="Twoje pytanie"' && exit 1)
	uv run rag ask "$(Q)" $(ARGS)

agent: ## agent z tool callingiem (interaktywnie albo Q="...")
	uv run rag agent $(if $(Q),"$(Q)",) $(ARGS)

eval: ## recall@k i MRR na złotym zbiorze (ARGS=--generate ocenia też odpowiedzi)
	uv run rag eval $(ARGS)

reset-cache: ## wykryj tryb structured output od nowa (np. po aktualizacji Ollamy)
	uv run rag reset-cache

test: ## testy jednostkowe (bez sieci)
	uv run pytest -q

lint: ## ruff
	uv run ruff check src tests
	uv run ruff format --check src tests
