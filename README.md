<div align="center">

# ♟️ Chess-AI: Deep Residual Neural Network Engine & Web Arena

### *Think you can outsmart a 10-block Deep Residual Neural Network?* 🔥

**Chess-AI** brings the power of **AlphaZero** straight to your desktop and browser with an interactive **Web Arena**! Powered by PyTorch and ONNX Runtime INT8 quantization, it features an interactive **Web GUI**, a **CLI terminal interface**, dual policy-guided search algorithms, and automated match gauntlets.

<br/>

[![Try Live Demo](https://img.shields.io/badge/🎮%20Try%20Live%20Demo-chess--ai--ml.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://chess-ai-ml.streamlit.app)

<br/>

[![CI](https://img.shields.io/badge/CI-Passing-2ea44f.svg?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/kem85/Chess-Ai/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-INT8%20Quantized-005CED.svg?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
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

# 2. Install all required dependencies (PyTorch, ONNX Runtime, python-chess, Streamlit)
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

> 🎯 **Browser Access**: Automatically opens at **`http://127.0.0.1:8000`** (or `http://localhost:8501` under Streamlit). Simply click or drag pieces with your mouse, watch legal moves highlight, and challenge the AI!

---

## ✨ Key Features

- 🖱️ **Interactive SVG Chessboard**: High-definition Staunton vector piece set with move highlighting, drag-and-drop, and legal target rings.
- 📊 **Real-Time Evaluation Meter**: Dynamic visual eval bar tracking engine evaluation from $-1.0$ (Black advantage) to $+1.0$ (White advantage).
- 🧠 **Dual-Head ResNet Engine**: 10 residual blocks (128 channels) trained to predict move policies and state values simultaneously.
- 🌲 **Dynamic Search Selector**: Switch between **Alpha-Beta Minimax** (with policy move ordering) and **AlphaZero MCTS** (PUCT simulations) in real time.
- 🔊 **Subtle Wooden Sound Synthesis**: Web Audio API audio effects for moves, captures, and check notifications with instant mute toggle.
- 🎮 **Full Match Controls**: Play as White, Black, or watch **AI Self-Play** with instant **Undo**, **Flip Board**, and **PGN Export**.
- ⚡ **ONNX INT8 Acceleration**: Quantized model weight footprint reduced by 75% (106 MB → 26 MB) for ultra-fast CPU/GPU inference.

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
# Option 1: Play as White against Minimax search at Lookahead Depth 3
python play.py --color white --engine minimax --depth 3

# Option 2: Play as Black against MCTS engine with 200 PUCT simulations
python play.py --color black --engine mcts --simulations 200
```

### Automated Benchmark Duels (`benchmark.py`)

Run automated self-play gauntlets and export formatted match PGN records to `pgn_exports/`:

```bash
# Run a 10-game automated self-play duel and export match PGN records
python benchmark.py --games 10 --depth 3
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