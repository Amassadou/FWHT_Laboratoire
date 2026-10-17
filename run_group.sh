#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:-}"
if [[ -z "$GROUP" ]]; then
  echo "Usage: $0 {geometry|compression|zoom|photometric} [max-images]" >&2
  exit 2
fi
if [[ -n "${2:-}" ]]; then
  export MAX_IMAGES="$2"
fi

case "$GROUP" in
  geometry|compression|zoom|photometric) ;;
  *) echo "Unknown group: $GROUP (choose geometry, compression, zoom, photometric)" >&2; exit 2 ;;
esac
"$LAB_DIR/run_${GROUP}.sh"
