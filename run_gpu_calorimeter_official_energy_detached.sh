#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_official/energy_full"
LOG_FILE="$OUTPUT_DIR/run.log"
PID_FILE="$OUTPUT_DIR/run.pid"
UNIT_NAME="calochallenge-official-energy"

mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/summary.json" ]; then
  printf '%s\n' "Official energy run is already complete."
  printf '%s\n' "Summary: $OUTPUT_DIR/summary.json"
  exit 0
fi

if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Official energy run is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached official energy run at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge official size-matched energy model" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_energy.py" \
  --output-dir "$OUTPUT_DIR" \
  --train-size 119790 \
  --validation-size 1210 \
  --iterations 250000 \
  --batch-size 256 \
  --validation-batch-size 1210 \
  --validation-every 4000 \
  --learning-rate 0.0001 \
  --weight-decay 0.1 \
  --hidden-dim 676 \
  --hidden-layers 5 \
  --mixture-components 16 \
  --seed 20261002 \
  --device cuda

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached official energy run started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"
