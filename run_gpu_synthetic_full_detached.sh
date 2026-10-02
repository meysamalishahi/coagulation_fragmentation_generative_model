#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_synthetic_full"
LOG_FILE="$OUTPUT_DIR/run.log"
PID_FILE="$OUTPUT_DIR/run.pid"
UNIT_NAME="coagulation-gpu-synthetic-full"

mkdir -p "$OUTPUT_DIR"

if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Full synthetic run is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)

printf '%s\n' "Starting detached full synthetic run at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="Coagulation continuous synthetic GPU benchmark" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_synthetic_benchmark.py" \
  --output-dir "$OUTPUT_DIR" \
  --seed 20261001 \
  --train-size 4000 \
  --validation-size 800 \
  --test-size 1000 \
  --epochs 30 \
  --batch-size 64 \
  --samples 1000 \
  --hidden-dim 128 \
  --transformer-layers 3 \
  --max-components 20

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"

printf '%s\n' "Detached full synthetic run started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "PID: $PID_FILE"
printf '%s\n' "Log: $LOG_FILE"
