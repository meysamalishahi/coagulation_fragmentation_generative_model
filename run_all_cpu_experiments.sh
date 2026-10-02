#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements-cpu.txt
fi

export MPLCONFIGDIR="$PROJECT_DIR/.mplconfig"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"

python3 cpu_validation.py --output-dir results/cpu_validation
python3 -m unittest -v test_cpu_validation.py
.venv/bin/python -m unittest -v test_cpu_extended.py
.venv/bin/python cpu_extended_experiments.py --output-dir results/cpu_extended

printf '%s\n' "CPU experiments complete."
printf '%s\n' "Correctness: results/cpu_validation/cpu_validation_summary.json"
printf '%s\n' "Report: results/cpu_extended/REPORT.md"
