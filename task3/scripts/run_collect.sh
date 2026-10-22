#!/usr/bin/env bash
# Full (or partial) real-answer sweep on the five PPAI agents.
# Usage:
#   ./scripts/run_collect.sh smoke          # 32 MMLU items, Qwen only
#   ./scripts/run_collect.sh open MMLU      # all non-gated agents on MMLU
#   ./scripts/run_collect.sh open all       # all open agents × all clusters
#   ./scripts/run_collect.sh A5_llama ARC-C # gated Llama (needs HF_TOKEN)
set -euo pipefail
cd "$(dirname "$0")/.."
ENV="${PPAI_ENV:-ppai}"
AGENT="${1:-smoke}"
CLUSTERS="${2:-MMLU}"
DEVICE="${DEVICE:-cuda:0}"
BACKEND="${BACKEND:-hf}"

run() {
  echo "+ conda run -n $ENV python -m collect.collect_answers $*"
  conda run --no-capture-output -n "$ENV" python -m collect.collect_answers "$@"
}

case "$AGENT" in
  smoke)
    run --agent A1_qwen --clusters MMLU --limit 32 --device "$DEVICE" --backend "$BACKEND" --batch-size 4
    ;;
  open)
    run --agent open --clusters "$CLUSTERS" --device "$DEVICE" --backend "$BACKEND" --batch-size 4
    ;;
  all)
    run --agent all --clusters "$CLUSTERS" --device "$DEVICE" --backend "$BACKEND" --batch-size 4
    ;;
  *)
    run --agent "$AGENT" --clusters "$CLUSTERS" --device "$DEVICE" --backend "$BACKEND" --batch-size 4
    ;;
esac

conda run -n "$ENV" python -m collect.status
