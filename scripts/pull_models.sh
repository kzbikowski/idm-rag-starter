#!/usr/bin/env sh
# Pobiera modele Ollamy wskazane w .env (LLM_MODEL i EMBED_MODEL).
set -e
[ -f .env ] && . ./.env
LLM_MODEL=${LLM_MODEL:-qwen3.5:4b}
EMBED_MODEL=${EMBED_MODEL:-qwen3-embedding:0.6b}
for model in "$LLM_MODEL" "$EMBED_MODEL"; do
  echo "==> ollama pull $model"
  ollama pull "$model"
done
