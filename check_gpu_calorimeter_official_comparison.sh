#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if systemctl --user is-active --quiet calochallenge-official-supervisor; then
  SUPERVISOR_PID=$(systemctl --user show --property=MainPID --value calochallenge-official-supervisor)
  printf '%s\n' "supervisor status=running service=calochallenge-official-supervisor pid=$SUPERVISOR_PID"
else
  SUPERVISOR_STATUS=$(systemctl --user show --property=ActiveState --value calochallenge-official-supervisor 2>/dev/null || true)
  printf '%s\n' "supervisor status=${SUPERVISOR_STATUS:-inactive} service=calochallenge-official-supervisor"
fi
if [ -f "$PROJECT_DIR/results/gpu_official/fair_comparison/supervisor_status.json" ]; then
  cat "$PROJECT_DIR/results/gpu_official/fair_comparison/supervisor_status.json"
fi
if [ -f "$PROJECT_DIR/results/gpu_official/fair_comparison/supervisor_failure.json" ]; then
  printf '%s\n' "supervisor failure detected:"
  cat "$PROJECT_DIR/results/gpu_official/fair_comparison/supervisor_failure.json"
fi

check_stage() {
  STAGE=$1
  UNIT_NAME=$2
  RESULT_FILE=$3
  LOG_FILE=$4
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
    printf '%s\n' "$STAGE status=running service=$UNIT_NAME pid=$RUN_PID"
  else
    STATUS=$(systemctl --user show --property=ActiveState --value "$UNIT_NAME" 2>/dev/null || true)
    printf '%s\n' "$STAGE status=${STATUS:-inactive} service=$UNIT_NAME"
  fi
  if [ -f "$RESULT_FILE" ]; then
    printf '%s\n' "$STAGE result=complete artifact=$RESULT_FILE"
  else
    printf '%s\n' "$STAGE result=incomplete"
  fi
  if [ -f "$LOG_FILE" ]; then
    tail -n 6 "$LOG_FILE"
  fi
}

check_stage \
  shape \
  calochallenge-official-shape \
  "$PROJECT_DIR/results/gpu_official/shape_full/summary.json" \
  "$PROJECT_DIR/results/gpu_official/shape_full/run.log"
check_stage \
  generation \
  calochallenge-official-generation \
  "$PROJECT_DIR/results/gpu_official/fair_comparison/generation_summary.json" \
  "$PROJECT_DIR/results/gpu_official/fair_comparison/generation.log"
check_stage \
  evaluation \
  calochallenge-official-evaluation \
  "$PROJECT_DIR/results/gpu_official/fair_comparison/paper_evaluation/summary.json" \
  "$PROJECT_DIR/results/gpu_official/fair_comparison/evaluation.log"
