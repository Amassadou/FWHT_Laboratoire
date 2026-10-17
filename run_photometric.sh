#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="python3"
[[ -x "$LAB_DIR/.venv/bin/python" ]] && DEFAULT_PYTHON="$LAB_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
MAX_IMAGES="${MAX_IMAGES:-1}"

"$PYTHON_BIN" "$LAB_DIR/code/benchmark_watermark_robustness.py" \
  --data-dir "$LAB_DIR/dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/pigs" \
  --max-images "$MAX_IMAGES" \
  --target-psnr 65 \
  --block-size 4 \
  --redundancy 3 \
  --group photometric \
  --attacks clean,brightness_080,brightness_120,contrast_080,contrast_120,saturation_050,saturation_150,grayscale,gamma_080,gamma_120,noise_sigma_3,noise_sigma_8,noise_sigma_15,blur_1_0,blur_2_0,sharpen \
  --output "$LAB_DIR/reports/photometric_davis.json"
