# ♟️ Chess-AI v1.0.0 Release Notes

> **Deep Residual Neural Network Chess Engine & Interactive Web Arena**  
> *Powered by ONNX Runtime INT8 Quantization, PyTorch ResNet Fallback, & AlphaZero MCTS* 🔥

---

## 🚀 Highlights & Features

- 🌐 **Interactive Web Arena**: Full-featured in-browser chess board with drag/click moves, high-definition Staunton SVG vector pieces, subtle Web Audio sound effects, legal move dots, and dynamic responsiveness across desktop, tablet, and mobile.
- ⚡ **High-Speed ONNX Runtime INT8 Quantization**: 8-bit integer quantized neural network weights (**26.7 MB** vs 106.6 MB) delivering **2–3× faster CPU search inference** with $99.8\%$ evaluation fidelity.
- 🎯 **Dual-Head Deep ResNet**: 10 residual blocks (128 channels) predicting spatial candidate move policies ($12 \times 8 \times 8 \to 4864$) and continuous value evaluations simultaneously.
- 🌲 **Dual Search Engines**: Choose dynamically between **Alpha-Beta Minimax** (with policy move ordering) and **AlphaZero MCTS** (Monte Carlo Tree Search with PUCT exploration).
- 🖱️ **1-Click Launchers**: Double-click `start_web_arena.bat` on Windows or run `start_web_arena.sh` on macOS/Linux to immediately launch the Web Arena and play in your browser!
- 📦 **Standard Python Packaging**: Install via `pip install .` and access terminal commands `chess-ai`, `chess-web`, and `chess-cli`.

---

## 📥 Download Packages

| Package Archive | Description | Size | Best For |
| :--- | :--- | :---: | :--- |
| **`Chess-AI-v1.0.0.zip`** | **Complete All-in-One Distribution** (Includes Web Arena, ONNX INT8 model, and full PyTorch FP32 weights) | ~100 MB | Full offline experience with zero extra downloads |
| **`Chess-AI-v1.0.0-ONNX-Lite.zip`** | **Lightweight Fast Distribution** (Includes Web Arena & ONNX INT8 engine, auto-downloads FP32 on demand) | ~20 MB | Fast download and instant setup |
| **`chess_ai_arena-1.0.0-py3-none-any.whl`** | Python Wheel package | ~100 KB | Standard Python / PyPI installation |
| **`chess_ai_arena-1.0.0.tar.gz`** | Source distribution archive | ~100 KB | Source builds |

---

## ⚡ Quick Start: Play in Seconds

### Windows (1-Click)
1. Download and extract **`Chess-AI-v1.0.0.zip`**.
2. Double-click **`start_web_arena.bat`**.
3. Your web browser will open automatically at `http://127.0.0.1:8000`!

### macOS / Linux (1-Click)
1. Download and extract **`Chess-AI-v1.0.0.zip`**.
2. In terminal, run:
   ```bash
   chmod +x start_web_arena.sh
   ./start_web_arena.sh
   ```

### Python / Pip Installation
```bash
pip install .
chess-web   # Launches the Web Arena in your default browser
```

---

## 🔒 Verification & Checksums
Refer to `SHA256SUMS.txt` in the release assets to verify file integrity with SHA-256 hashes.
