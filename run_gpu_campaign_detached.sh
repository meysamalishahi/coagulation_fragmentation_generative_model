#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_campaign"
LOG_FILE="$OUTPUT_DIR/campaign.log"
PID_FILE="$OUTPUT_DIR/campaign.pid"
UNIT_NAME="coagulation-gpu-campaign"

mkdir -p "$OUTPUT_DIR"

if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "GPU campaign is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached GPU campaign at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="Coagulation multi-seed GPU experiment campaign" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_experiment_campaign.py" run --root "$OUTPUT_DIR"

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached GPU campaign started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Progress: $OUTPUT_DIR/campaign_progress.json"
printf '%s\n' "Log: $LOG_FILE"
