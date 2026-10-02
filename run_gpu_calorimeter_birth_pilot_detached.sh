#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_campaign/real/calochallenge_ds1_photons_birth_only_pilot"
LOG_FILE="$OUTPUT_DIR/run.log"
PID_FILE="$OUTPUT_DIR/run.pid"
UNIT_NAME="calochallenge-birth-only-pilot"

mkdir -p "$OUTPUT_DIR"

if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Birth-only pilot is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached birth-only pilot at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge capacity-matched birth-only pilot" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_benchmark.py" \
  --output-dir "$OUTPUT_DIR" \
  --model-type birth_only \
  --train-size 1000 \
  --validation-size 200 \
  --test-size 200 \
  --epochs 8 \
  --batch-size 32 \
  --samples 200 \
  --data-components 32 \
  --max-components 48 \
  --hidden-dim 128 \
  --transformer-layers 3 \
  --mixture-components 8 \
  --learning-rate 0.0002 \
  --sample-batch-size 8 \
  --sample-checkpoint-every 25 \
  --seed 20261001 \
  --device cuda

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached birth-only pilot started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"

