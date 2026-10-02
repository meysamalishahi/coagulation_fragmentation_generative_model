#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_official/fair_comparison"
SAMPLE_FILE="$OUTPUT_DIR/generated_ds1_photons.hdf5"
EVAL_DIR="$OUTPUT_DIR/paper_evaluation"
LOG_FILE="$OUTPUT_DIR/evaluation.log"
PID_FILE="$OUTPUT_DIR/evaluation.pid"
UNIT_NAME="calochallenge-official-evaluation"

if [ ! -f "$OUTPUT_DIR/generation_summary.json" ] || [ ! -f "$SAMPLE_FILE" ]; then
  printf '%s\n' "Refusing evaluation: independent official generation is incomplete." >&2
  exit 1
fi
if [ -f "$EVAL_DIR/summary.json" ]; then
  printf '%s\n' "Paper-exact evaluation is already complete."
  printf '%s\n' "Summary: $EVAL_DIR/summary.json"
  exit 0
fi
if systemctl --user is-active --quiet "$UNIT_NAME"; then
  ACTIVE_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Official evaluation is already active (service $UNIT_NAME, PID $ACTIVE_PID)."
  exit 0
fi

CONDA_BIN=$(command -v conda)
mkdir -p "$OUTPUT_DIR"
printf '%s\n' "Starting detached paper-exact evaluation at $(date -Iseconds)" >>"$LOG_FILE"
systemd-run --user \
  --unit="$UNIT_NAME" \
  --collect \
  --description="CaloChallenge paper-exact AUC KPD FPD evaluation" \
  --property="Type=exec" \
  --property="WorkingDirectory=$PROJECT_DIR" \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONUNBUFFERED=1" \
  "$CONDA_BIN" run --no-capture-output -n tabkde \
  python "$PROJECT_DIR/gpu_calorimeter_evaluate_paper.py" \
  --sample-file "$SAMPLE_FILE" \
  --reference-file "$PROJECT_DIR/data/calochallenge/dataset_1_photons/dataset_1_photons_2.hdf5" \
  --geometry "$PROJECT_DIR/external/calochallenge-homepage/code/binning_dataset_1_photons.xml" \
  --output-dir "$EVAL_DIR" \
  --cut-mev 0.015 \
  --seed 20261005

RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
printf '%s\n' "$RUN_PID" >"$PID_FILE"
printf '%s\n' "Detached paper-exact evaluation started (service $UNIT_NAME, PID $RUN_PID)."
printf '%s\n' "Log: $LOG_FILE"
