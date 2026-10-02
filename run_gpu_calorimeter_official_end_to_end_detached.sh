#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_official/fair_comparison"
LOG_FILE="$OUTPUT_DIR/supervisor.log"
PID_FILE="$OUTPUT_DIR/supervisor.pid"
UNIT_NAME="calochallenge-official-supervisor"

mkdir -p "$OUTPUT_DIR"
if [ -f "$OUTPUT_DIR/final_summary.json" ] && [ -f "$OUTPUT_DIR/FINAL_REPORT.md" ]; then
  printf '%s\n' "End-to-end official comparison is already complete."
  printf '%s\n' "Report: $OUTPUT_DIR/FINAL_REPORT.md"
  exit 0
fi
if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "End-to-end supervisor is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached end-to-end supervisor at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge end-to-end fair-comparison supervisor" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_official_supervisor.py"

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached end-to-end supervisor started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"
