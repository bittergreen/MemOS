#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common parameters for all scripts
LIB="memos-api"
VERSION="202606_benchmarking_v1"
WORKERS=10
TOPK=20
ASYNC_MODE="async"  # Options: sync, async. Empty means backend default.

# Optional: space-separated views (each token is one CLI arg).
# Leave empty to omit allow_memory_view from the add API (server default).
# Example: ALLOW_MEMORY_VIEW="detail_factual preference event"
ALLOW_MEMORY_VIEW=""

INGESTION_EXTRA_ARGS=""
if [ -n "$ALLOW_MEMORY_VIEW" ]; then
    INGESTION_EXTRA_ARGS="$INGESTION_EXTRA_ARGS --allow-memory-view $ALLOW_MEMORY_VIEW"
fi
if [ -n "$ASYNC_MODE" ]; then
    INGESTION_EXTRA_ARGS="$INGESTION_EXTRA_ARGS --async-mode $ASYNC_MODE"
fi

echo "Running locomo_ingestion.py..."
if ! CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/locomo/locomo_ingestion.py" --lib "$LIB" --version "$VERSION" --workers "$WORKERS" $INGESTION_EXTRA_ARGS; then
    echo "Error running locomo_ingestion.py"
    exit 1
fi

echo "Running locomo_search.py..."
if ! CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/locomo/locomo_search.py" --lib "$LIB" --version "$VERSION" --top_k "$TOPK" --workers "$WORKERS"; then
    echo "Error running locomo_search.py"
    exit 1
fi

echo "Running locomo_responses.py..."
if ! python "$SCRIPT_DIR/locomo/locomo_responses.py" --lib "$LIB" --version "$VERSION"; then
    echo "Error running locomo_responses.py."
    exit 1
fi

echo "Running locomo_eval.py..."
if ! python "$SCRIPT_DIR/locomo/locomo_eval.py" --lib "$LIB" --version "$VERSION" --workers "$WORKERS" --num_runs 3; then
    echo "Error running locomo_eval.py"
    exit 1
fi

echo "Running locomo_metric.py..."
if ! python "$SCRIPT_DIR/locomo/locomo_metric.py" --lib "$LIB" --version "$VERSION"; then
    echo "Error running locomo_metric.py"
    exit 1
fi

echo "All scripts completed successfully!"
