#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$LAB_DIR/.venv"
"$LAB_DIR/.venv/bin/python" -m pip install --upgrade pip
"$LAB_DIR/.venv/bin/pip" install -r "$LAB_DIR/requirements-experiments.txt"

"$LAB_DIR/.venv/bin/python" - <<'PY'
import cv2
print(f"Environnement prêt : OpenCV {cv2.__version__}, SIFT={hasattr(cv2, 'SIFT_create')}")
PY
