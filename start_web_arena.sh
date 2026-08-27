#!/usr/bin/env bash
# =====================================================================
#   ♟️  CHESS-AI: DEEP RESIDUAL NEURAL ENGINE & WEB ARENA
#   ⚡  Powered by ONNX Runtime INT8 Quantization & AlphaZero MCTS
# =====================================================================

set -e

echo "====================================================================="
echo "  ♟️  CHESS-AI: DEEP RESIDUAL NEURAL ENGINE & WEB ARENA"
echo "  ⚡  Powered by ONNX Runtime INT8 Quantization & AlphaZero MCTS"
echo "====================================================================="
echo ""

# Find suitable python binary
if command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo "[ERROR] Python 3.10+ is required but was not found in PATH."
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

echo "[*] Using Python: $($PY_BIN --version)"
echo "[*] Launching Chess-AI Web Arena on http://127.0.0.1:8000 ..."
echo "[*] Press Ctrl+C to terminate the server."
echo ""

$PY_BIN app.py || {
    echo ""
    echo "[!] Encountered an error. Checking/installing requirements..."
    $PY_BIN -m pip install -r requirements.txt
    echo "[*] Retrying launch..."
    $PY_BIN app.py
}
