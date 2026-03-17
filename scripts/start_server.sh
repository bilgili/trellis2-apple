#!/bin/bash
# Start the Trellis2 MLX API server on macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Set macOS-specific environment
export SPARSE_CONV_BACKEND=pytorch
export ATTN_BACKEND=sdpa
export PYTORCH_ENABLE_MPS_FALLBACK=1  # deform_conv2d in RMBG not yet on MPS
export PYTHONPATH="$ROOT_DIR/o-voxel:${PYTHONPATH:-}"

# Defaults
WEIGHTS="${TRELLIS2_WEIGHTS:-weights/TRELLIS.2-4B}"
PORT="${TRELLIS2_PORT:-8082}"

echo "Starting Trellis2 MLX API server..."
echo "  Weights: $WEIGHTS"
echo "  Port: $PORT"

python api_server.py --weights "$WEIGHTS" --port "$PORT"
