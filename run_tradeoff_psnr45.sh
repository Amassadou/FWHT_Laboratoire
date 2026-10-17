#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$LAB_DIR/code/experiment_fwht_calibration.py" \
  --data-dir "$LAB_DIR/dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/pigs" \
  --max-images 3 \
  --block-sizes 4 \
  --redundancies 3 \
  --target-psnr 45 \
  --attacks clean,jpeg_q90,jpeg_q70,jpeg_q50,crop_90,crop_75,rotate_5,rotate_15,rotate_90 \
  --output "$LAB_DIR/reports/geometric_compression_robust_davis.json"
