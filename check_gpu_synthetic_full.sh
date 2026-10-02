#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_synthetic_full"
LOG_FILE="$OUTPUT_DIR/run.log"
PID_FILE="$OUTPUT_DIR/run.pid"
UNIT_NAME="coagulation-gpu-synthetic-full"

if [ ! -f "$PID_FILE" ]; then
  printf '%s\n' "No full-run PID file exists."
  exit 1
fi

RUN_PID=$(sed -n '1p' "$PID_FILE")
case "$RUN_PID" in
  ''|*[!0-9]*)
    printf '%s\n' "Invalid PID file: $PID_FILE"
    exit 1
    ;;
esac

if systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
  printf '%s\n' "Full synthetic run is active (service $UNIT_NAME, PID $RUN_PID)."
  systemctl --user status "$UNIT_NAME" --no-pager --lines=5
elif [ -f "$OUTPUT_DIR/summary.json" ]; then
  printf '%s\n' "Full synthetic run is complete."
elif kill -0 "$RUN_PID" 2>/dev/null; then
  printf '%s\n' "Full synthetic run process is active (PID $RUN_PID)."
  ps -o pid,ppid,sid,stat,etime,cmd -p "$RUN_PID"
else
  printf '%s\n' "Full synthetic run is not active and has no completed summary (last PID $RUN_PID)."
fi

if [ -f "$LOG_FILE" ]; then
  printf '%s\n' "Recent log output:"
  tail -n 20 "$LOG_FILE"
fi
