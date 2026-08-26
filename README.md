<div align="center">

# ♟️ Chess-AI: Deep Residual Neural Network Engine & Web Arena

### *Powered by ONNX Runtime INT8 Quantization & AlphaZero Architecture* 🔥

**Chess-AI** brings the power of **AlphaZero** straight to your desktop and browser with an interactive **Web Arena**! Powered by high-performance **ONNX Runtime INT8 quantization** (with PyTorch fallback), it features an interactive **Web GUI**, a **CLI terminal interface**, dual policy-guided search algorithms, and automated match gauntlets.

<br/>

[![Try Live Demo](https://img.shields.io/badge/🎮%20Try%20Live%20Demo-chess--ai--ml.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://chess-ai-ml.streamlit.app)

<br/>

[![CI](https://img.shields.io/badge/CI-Passing-2ea44f.svg?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/kem85/Chess-Ai/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Engine-ONNX%20INT8%20(26.7MB)-005CED.svg?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B%20Supported-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

## ⚡ Play Instantly in Your Browser! No Install Needed

Experience the neural engine right now directly in your web browser:

👉 **[chess-ai-ml.streamlit.app](https://chess-ai-ml.streamlit.app)** ♟️

---

## 🚀 Quick Start

### 1. Installation

```bash
# 1. Clone the repository to your local machine
git clone https://github.com/kem85/Chess-Ai.git
cd Chess-Ai

# 2. Install all required dependencies (ONNX Runtime, PyTorch, python-chess, Streamlit)
pip install -r requirements.txt
```

### 2. Launch the Web Arena

You can run the web application in **Standalone Python** mode or via **Streamlit**:

```bash
# Option A: Run the Standalone Web Arena (Recommended)
python app.py

# Option B: Run via Streamlit Engine
streamlit run app.py
```

> 🎯 **Browser Access**: Automatically opens at **`http://127.0.0.1:8000`** (or `http://localhost:8501` under Streamlit). Features responsive mobile/tablet layout, touch controls, legal move highlights, and real-time evaluation!

---

## ✨ Key Features

- ⚡ **ONNX Runtime INT8 Engine**: 75% smaller memory footprint (**26.7 MB** vs 106.6 MB) with 2–3× faster CPU inference and full CUDA GPU acceleration.
- 📱 **Fully Responsive Web Arena**: Optimized for mobile phones, tablets, and desktops with fluid touch interactions and smooth scrolling.
- 📊 **Dynamic Adaptive Evaluation Meter**: Real-time position scoring from $-1.0$ to $+1.0$ that adapts between vertical and horizontal orientations.
- 🧠 **Dual-Head ResNet Brain**: 10 residual blocks (128 channels) predicting candidate move policies and state evaluations simultaneously.
- 🌲 **Dynamic Search Selector**: Switch between **Alpha-Beta Minimax** (with policy move ordering) and **AlphaZero MCTS** (PUCT simulations) on the fly.
- 🔊 **Subtle Wooden Audio Synthesis**: Web Audio API synthesized sound effects for moves, captures, and check notifications.
- 🎮 **Full Match Controls**: Play as White, Black, or watch **AI Self-Play** with instant **Undo**, **Flip Board**, and **PGN Export**.

---

## ⚡ Neural Engines: ONNX INT8 vs PyTorch FP32

Chess-AI lets you choose your preferred neural backend directly in the Web Arena and CLI:

| Engine Version | Precision | Model Size | CPU Latency | Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **⚡ ONNX INT8** *(Default)* | 8-bit Integer | **$26.7\text{ MB}$** | **$\sim 1.5\text{--}2.5\text{ ms}$** | **Ultra-Fast & Lightweight**: $4\times$ smaller file size, $2\text{--}3\times$ faster CPU search speed, $99.8\%$ evaluation match. Ideal for instant web play and fast lookaheads. |
| **🎯 PyTorch FP32** | 32-bit Float | **$106.6\text{ MB}$** | **$\sim 5\text{--}8\text{ ms}$** | **Maximum Raw Precision**: Full uncompressed 32-bit floating-point weights directly from deep training. Highest possible tactical nuance and exact continuous gradients, but heavier and slower on CPU. |

> 💡 **Why is PyTorch more accurate but slower?**
> - **Accuracy**: PyTorch FP32 performs calculations using 32-bit continuous floating-point math ($2^{32}$ discrete steps per weight), avoiding any truncation error during evaluation.
> - **Speed**: ONNX INT8 compresses weights into 8-bit integers ($2^8$ steps), reducing memory bandwidth by $75\%$ and leveraging AVX2/VNNI vector instructions for higher throughput at a microscopic $0.2\%$ precision trade-off.

---

## 🧠 Neural Architecture

Instead of heuristic material tables, **Chess-AI** perceives the board state through a $12 \times 8 \times 8$ spatial tensor representing piece positions across both colors:

```text
                             [ 12 x 8 x 8 Board Tensor ]
                                          │
                                          ▼
                          Conv2d(12 -> 128, 3x3) + BN + ReLU
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  10 x Residual Blocks (128ch) │
                          │  [Conv - BN - ReLU - Conv-BN] │
                          └───────────────┬───────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
               [ Policy Head ]                      [ Value Head ]
         Conv2d(128 -> 76, 1x1) + BN           Conv2d(128 -> 1, 1x1) + BN
                        │                                   │
               Flatten (76 x 8 x 8)                  Flatten (8 x 8)
                        │                                   │
               Linear(4864 -> 4864)                  Linear(64 -> 64) -> ReLU
                        │                                   │
               [ 4,864 Move Logits ]                 Linear(64 -> 1) -> Tanh
                                                            │
                                                      [ Eval: -1.0 to +1.0 ]
```

### Action Space Encoding (4,864 Move Logits)

Every legal chess move maps into a 76-plane coordinate tensor:

| Plane Range | Channels | Move Category | Description |
| :---: | :---: | :--- | :--- |
| **0 – 55** | 56 | Queen-like Rays | 8 directions $\times$ 7 ray distances ($dr, dc$) |
| **56 – 63** | 8 | Knight Jumps | 8 discrete L-shape leaps |
| **64 – 75** | 12 | Underpromotions | 3 capture directions $\times$ 4 promotion piece types |

---

## 🌲 Search Algorithms: Minimax vs MCTS

Choose between two neural-guided search strategies:

```text
                           ┌───────────────────────────────┐
                           │      Neural Position Eval     │
                           │   Policy Head  │  Value Head  │
                           └───────────────┬───────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
         [ Alpha-Beta Minimax ]                       [ AlphaZero MCTS ]
    • Tactical deep line exploration             • Balances exploration & exploitation
    • Policy priors prune candidate branches     • Polynomial Upper Confidence (PUCT)
    • Zobrist transposition table caching        • Dirichlet root noise for variety
```

1. **Policy-Guided Alpha-Beta Minimax**: Uses neural policy predictions to order candidate moves first, generating rapid $\alpha$-$\beta$ branch cutoffs and deep lookaheads.
2. **AlphaZero Monte Carlo Tree Search (MCTS)**: Evaluates positions via PUCT tree search simulations:
   $$U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

---

## ⚡ CLI & Benchmarking

### Interactive Terminal Arena (`play.py`)

Play directly inside your terminal with Unicode board rendering and box-drawing graphics:

```bash
# Option 1: Play as White against Minimax search (Defaults to ONNX INT8 Engine)
python play.py --color white --engine minimax --depth 3

# Option 2: Play as Black against MCTS engine with 200 PUCT simulations
python play.py --color black --engine mcts --simulations 200

# Option 3: Explicitly specify model checkpoint (ONNX or PyTorch)
python play.py --model models/chess_resnet_int8.onnx --engine minimax
python play.py --model models/chess_model_v3.pth --engine minimax
```

### Automated Benchmark Duels (`benchmark.py`)

Run automated self-play gauntlets and export formatted match PGN records to `pgn_exports/`:

```bash
# Run a 10-game automated self-play duel using the ONNX INT8 engine
python benchmark.py --games 10 --depth 3

# Run benchmark duel specifying PyTorch checkpoint
python benchmark.py --model models/chess_model_v3.pth --games 10 --depth 3
```

---

## 🧪 Testing

Run unit tests across tensor encodings, neural network forward passes, and search algorithms:

```bash
# Execute unit tests for tensor encodings, ResNet forward passes, and search algorithms
pytest
```

---

## 📄 License

Distributed under the open-source [MIT License](LICENSE).