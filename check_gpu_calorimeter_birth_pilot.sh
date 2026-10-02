#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_campaign/real/calochallenge_ds1_photons_birth_only_pilot"
LOG_FILE="$OUTPUT_DIR/run.log"
UNIT_NAME="calochallenge-birth-only-pilot"

if systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
  RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "Birth-only pilot is active (service $UNIT_NAME, PID $RUN_PID)."
  systemctl --user status "$UNIT_NAME" --no-pager --lines=4
elif [ -f "$OUTPUT_DIR/summary.json" ]; then
  printf '%s\n' "Birth-only pilot is complete. Report: $OUTPUT_DIR/REPORT.md"
else
  printf '%s\n' "Birth-only pilot is not active and no final result exists."
fi

if [ -f "$LOG_FILE" ]; then
  printf '%s\n' "Recent birth-only pilot log:"
  tail -n 25 "$LOG_FILE"
fi

