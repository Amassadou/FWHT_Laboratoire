#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="python3"
[[ -x "$LAB_DIR/.venv/bin/python" ]] && DEFAULT_PYTHON="$LAB_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
MAX_IMAGES="${MAX_IMAGES:-1}"

"$PYTHON_BIN" "$LAB_DIR/code/benchmark_watermark_robustness.py" \
  --data-dir "$LAB_DIR/dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/pigs" \
  --max-images "$MAX_IMAGES" --target-psnr 65 --block-size 4 --redundancy 3 \
  --group zoom \
  --attacks clean,resize_75,resize_50,zoom_125,zoom_150 \
  --output "$LAB_DIR/reports/zoom_davis.json"
