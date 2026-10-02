#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SHAPE_MODEL="$PROJECT_DIR/results/gpu_official/shape_full/model.pt"
SHAPE_SUMMARY="$PROJECT_DIR/results/gpu_official/shape_full/summary.json"
OUTPUT_DIR="$PROJECT_DIR/results/gpu_official/fair_comparison"
OUTPUT_FILE="$OUTPUT_DIR/generated_ds1_photons.hdf5"
LOG_FILE="$OUTPUT_DIR/generation.log"
PID_FILE="$OUTPUT_DIR/generation.pid"
UNIT_NAME="calochallenge-official-generation"

if [ ! -f "$SHAPE_MODEL" ] || [ ! -f "$SHAPE_SUMMARY" ]; then
  printf '%s\n' "Refusing generation: final-step full shape training is incomplete." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
if [ -f "$OUTPUT_DIR/generation_summary.json" ]; then
  printf '%s\n' "Official generation is already complete."
  printf '%s\n' "Sample: $OUTPUT_FILE"
  exit 0
fi
if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Official generation is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached official generation at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge official independent shower generation" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_generate_official.py" \
  --output-file "$OUTPUT_FILE" \
  --energy-checkpoint "$PROJECT_DIR/results/gpu_official/energy_full/model.pt" \
  --shape-checkpoint "$SHAPE_MODEL" \
  --samples 121000 \
  --batch-size 100 \
  --checkpoint-every 1000 \
  --seed 20261004 \
  --device cuda

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached official generation started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"
