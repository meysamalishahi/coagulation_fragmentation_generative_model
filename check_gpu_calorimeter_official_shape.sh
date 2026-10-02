#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SMOKE_DIR="$PROJECT_DIR/results/gpu_official/shape_smoke"
FULL_DIR="$PROJECT_DIR/results/gpu_official/shape_full"

for KIND in smoke full; do
  if [ "$KIND" = smoke ]; then
    OUTPUT_DIR="$SMOKE_DIR"
    UNIT_NAME="calochallenge-official-shape-smoke"
  else
    OUTPUT_DIR="$FULL_DIR"
    UNIT_NAME="calochallenge-official-shape"
  fi
  LOG_FILE="$OUTPUT_DIR/run.log"
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
    printf '%s\n' "$KIND status=running service=$UNIT_NAME pid=$RUN_PID"
  else
    STATUS=$(systemctl --user show --property=ActiveState --value "$UNIT_NAME" 2>/dev/null || true)
    printf '%s\n' "$KIND status=${STATUS:-inactive} service=$UNIT_NAME"
  fi
  if [ -f "$OUTPUT_DIR/summary.json" ]; then
    printf '%s\n' "$KIND result=complete summary=$OUTPUT_DIR/summary.json"
  else
    printf '%s\n' "$KIND result=incomplete"
  fi
  if [ -f "$LOG_FILE" ]; then
    tail -n 8 "$LOG_FILE"
  fi
done
