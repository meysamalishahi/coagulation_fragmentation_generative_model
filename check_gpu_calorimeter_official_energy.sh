#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_official/energy_full"
LOG_FILE="$OUTPUT_DIR/run.log"
UNIT_NAME="calochallenge-official-energy"

if systemctl --user is-active --quiet "$UNIT_NAME"; then
  RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "status=running service=$UNIT_NAME pid=$RUN_PID"
else
  STATUS=$(systemctl --user show --property=ActiveState --value "$UNIT_NAME" 2>/dev/null || true)
  printf '%s\n' "status=${STATUS:-inactive} service=$UNIT_NAME"
fi

if [ -f "$OUTPUT_DIR/summary.json" ]; then
  printf '%s\n' "result=complete summary=$OUTPUT_DIR/summary.json"
else
  printf '%s\n' "result=incomplete"
fi

if [ -f "$LOG_FILE" ]; then
  tail -n 12 "$LOG_FILE"
fi
