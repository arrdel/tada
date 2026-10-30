#!/usr/bin/env bash
# Sequenced single-GPU pipeline runner:
#   1. If a job is already active on the target GPU, waits for it to finish.
#   2. Runs each queued stage back-to-back on that GPU so the next job starts
#      immediately after the previous one completes.
#
# Usage:
#   GPU=2 ./scripts/run_gpu_sequence.sh
#   tmux new-session -d -s seq_gpu2 "GPU=2 ./scripts/run_gpu_sequence.sh"

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
GPU="${GPU:-2}"
LOG="${LOG:-/tmp/gpu_sequence_gpu${GPU}.log}"

if [ -x "$HOME/miniconda3/envs/ppai/bin/python" ]; then
  PY="$HOME/miniconda3/envs/ppai/bin/python"
elif [ -x "$HOME/miniconda3/bin/python" ]; then
  PY="$HOME/miniconda3/bin/python"
else
  PY="python3"
fi

export CUDA_VISIBLE_DEVICES="$GPU"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU] $*" | tee -a "$LOG"
}

wait_for_active_gpu_procs() {
  local pids
  pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -n "$pids" ]; then
    log "Waiting for currently running process(es) on GPU $GPU: $pids"
    for pid in $pids; do
      while kill -0 "$pid" 2>/dev/null; do
        sleep 5
      done
    done
    log "Previous GPU $GPU job finished; starting queued sequence immediately."
  fi
}

run_stage() {
  local name="$1"
  shift
  log "=== START STAGE: $name ==="
  "$@" 2>&1 | tee -a "$LOG"
  log "=== DONE STAGE:  $name ==="
}

wait_for_active_gpu_procs

# Stage 1: Verify / complete any remaining real-answer traces across all 5 agents
run_stage "collect_verify_all_5_agents" \
  "$PY" -m collect.collect_answers --agent all --clusters all --device cuda:0 --backend hf --batch-size 12 --max-new-tokens 128

run_stage "trace_bank_status" \
  "$PY" -m collect.status

# Stage 2: GPU Actor-Critic training + progressive ablation for Trust + Confidence + Availability + Trainable Beta
run_stage "train_beta_confidence_availability" \
  "$PY" -m experiments.train_beta_confidence

# Stage 3: Full Task III experiment suite (robustness, cost, load, RL escalation, ablations)
run_stage "run_all_experiments" \
  "$PY" -m experiments.run_all

log "ALL SEQUENCED STAGES COMPLETED SUCCESSFULLY."
