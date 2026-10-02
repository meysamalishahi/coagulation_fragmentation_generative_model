#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE_RUN="$PROJECT_DIR/results/gpu_campaign/real/calochallenge_ds1_photons"
OUTPUT_DIR="$SOURCE_RUN/calibration"
LOG_FILE="$OUTPUT_DIR/run.log"
PID_FILE="$OUTPUT_DIR/run.pid"
UNIT_NAME="calochallenge-sampler-calibration"

mkdir -p "$OUTPUT_DIR"

if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Calibration is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached sampler calibration at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge validation-only sampler calibration" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_calibration.py" \
  --run-dir "$SOURCE_RUN" \
  --output-dir "$OUTPUT_DIR" \
  --calibration-samples 200 \
  --test-samples 1000 \
  --temperatures 0.7 1.0 1.3 \
  --rate-scales 1.0 1.5 2.0 3.0 \
  --sample-batch-size 8 \
  --checkpoint-every 25 \
  --device cuda

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached calibration started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"

