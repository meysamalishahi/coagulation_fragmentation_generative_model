#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SMOKE_SUMMARY="$PROJECT_DIR/results/gpu_official/shape_smoke/summary.json"
OUTPUT_DIR="$PROJECT_DIR/results/gpu_official/shape_full"
LOG_FILE="$OUTPUT_DIR/run.log"
PID_FILE="$OUTPUT_DIR/run.pid"
UNIT_NAME="calochallenge-official-shape"

if [ ! -f "$SMOKE_SUMMARY" ]; then
  printf '%s\n' "Refusing full training: the full-size shape smoke test has not completed." >&2
  printf '%s\n' "Run: sh run_gpu_calorimeter_official_shape_smoke_detached.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [ -f "$OUTPUT_DIR/summary.json" ]; then
  printf '%s\n' "Official shape run is already complete."
  printf '%s\n' "Summary: $OUTPUT_DIR/summary.json"
  exit 0
fi
if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Official shape run is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  printf '%s\n' "Log: $LOG_FILE"
  exit 0
fi

CONDA_BIN=$(command -v conda)
printf '%s\n' "Starting detached official shape run at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge official parameter-matched shape model" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_shape.py" \
  --output-dir "$OUTPUT_DIR" \
  --train-size 119790 \
  --validation-size 1210 \
  --iterations 800000 \
  --batch-size 64 \
  --validation-batch-size 64 \
  --validation-every 4000 \
  --validation-samples 256 \
  --learning-rate 0.0001 \
  --weight-decay 0.1 \
  --core-components 48 \
  --hidden-dim 480 \
  --transformer-layers 6 \
  --attention-heads 6 \
  --mixture-components 16 \
  --residual-hidden-dim 1536 \
  --residual-layers 4 \
  --seed 20261003 \
  --device cuda

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached official shape run started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"
