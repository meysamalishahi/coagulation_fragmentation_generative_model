#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$PROJECT_DIR/results/gpu_campaign"
LOG_FILE="$OUTPUT_DIR/campaign.log"
PID_FILE="$OUTPUT_DIR/campaign.pid"
UNIT_NAME="coagulation-gpu-campaign"

if systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
  RUN_PID=$(systemctl --user show --property=MainPID --value "$UNIT_NAME")
  printf '%s\n' "GPU campaign is active (service $UNIT_NAME, PID $RUN_PID)."
  systemctl --user status "$UNIT_NAME" --no-pager --lines=5
elif [ -f "$OUTPUT_DIR/campaign_progress.json" ] && grep -q '"stage": "complete"' "$OUTPUT_DIR/campaign_progress.json"; then
  printf '%s\n' "GPU campaign is complete. Report: $OUTPUT_DIR/REPORT.md"
elif [ -f "$PID_FILE" ]; then
  RUN_PID=$(sed -n '1p' "$PID_FILE")
  printf '%s\n' "GPU campaign is not active and has no final report (last PID $RUN_PID)."
else
  printf '%s\n' "GPU campaign has not been started."
fi

if [ -f "$OUTPUT_DIR/campaign_progress.json" ]; then
  printf '%s\n' "Current progress:"
  sed -n '1,20p' "$OUTPUT_DIR/campaign_progress.json"
fi
if [ -f "$LOG_FILE" ]; then
  printf '%s\n' "Recent campaign log:"
  tail -n 20 "$LOG_FILE"
fi

CURRENT_RUN_LOG=$(find "$OUTPUT_DIR" -mindepth 2 -type f -name run.log -printf '%T@ %p\n' 2>/dev/null \
  | sort -n | tail -n 1 | cut -d ' ' -f 2-)
if [ -n "$CURRENT_RUN_LOG" ]; then
  printf '%s\n' "Recent active-run log ($CURRENT_RUN_LOG):"
  tail -n 12 "$CURRENT_RUN_LOG"
fi
