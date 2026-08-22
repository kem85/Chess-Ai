<div align="center">

# ♟️ Chess-AI: Play Against a Neural Brain in Your Browser!

### *Think you can outsmart a 10-block Deep Residual Neural Network?* 🔥

**Chess-AI** brings the power of **AlphaZero** straight to your desktop with a **super-easy, highly interactive Web GUI**! No complex chess GUI setups, engine configurations, or compilation steps needed — simply run one command, and an interactive browser arena pops open ready for you to play, experiment, and analyze moves in real time.

<br/>

[![CI](https://img.shields.io/badge/CI-Passing-2ea44f.svg?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/kem85/Chess-Ai/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-INT8%20Quantized-005CED.svg?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg?style=flat-square)](https://opensource.org/licenses/MIT)

<br/>

[🚀 Jump In & Play](#-jump-in--play-in-seconds) • [✨ Interactive Features](#-interactive-features--ease-of-play) • [🧠 How the Brain Thinks](#-how-the-brain-thinks) • [🌲 The Search Engines](#-two-ways-to-think-minimax-vs-mcts) • [⚡ Lightning Fast Inference](#-lightning-fast-inference)

</div>

---

## 🚀 Jump In & Play in Seconds!

Getting started is effortless. One command fires up the neural engine and opens the interactive arena directly in your web browser:

```bash
# 1. Clone & install
git clone https://github.com/kem85/Chess-Ai.git
cd Chess-Ai
pip install -r requirements.txt

# 2. Launch the Interactive GUI!
python app.py
```

> 🎯 **Super Easy to Play**: The browser automatically opens to `http://localhost:8000`. Just click or drag pieces with your mouse, watch legal moves highlight, and challenge the AI!

---

## ✨ Interactive Features & Ease of Play

- 🖱️ **Effortless Drag-and-Drop / Click-to-Move**: Intuitive, responsive piece controls with glowing legal move dots and capture rings.
- 📊 **Live Dynamic Evaluation Bar**: Watch the neural network calculate its advantage in real time from $-1.0$ (Black winning) to $+1.0$ (White winning).
- 🧠 **Dual-Head Residual Backbone**: 10 residual blocks (128 channels) trained to evaluate complex board strategies and tactics.
- 🌲 **On-the-Fly Search Switcher**: Toggle smoothly between **Alpha-Beta Minimax** (with depth slider 1–6) and **AlphaZero MCTS** (with simulations slider 50–800).
- 🔊 **Subtle Wooden Sound FX**: Gentle acoustic sound effects with a convenient one-click **Mute/Unmute** button.
- 🎮 **Full Player Freedom**: Play as White, Black, or watch **AI vs AI Self-Play** with instant **Undo Move**, **Flip Board**, and **PGN Export**.

---

## 🧠 How the Brain Thinks

Instead of relying on hardcoded heuristics, **Chess-AI** perceives the entire chessboard through a high-dimensional spatial tensor, passing it through 10 Residual Blocks to simultaneously predict **what move to play (Policy)** and **who is winning (Value)**:

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

### The 76 Action Planes ($4,864$ Moves)
Every possible chess move (including knight jumps, queen rays, and underpromotions) maps into a discrete 76-plane coordinate grid:

| Plane Range | Channels | Move Category | Explanation |
| :---: | :---: | :--- | :--- |
| **0 – 55** | 56 | Queen-like Rays | 8 directions $\times$ 7 ray distances ($dr, dc$) |
| **56 – 63** | 8 | Knight Jumps | 8 discrete L-shape leaps |
| **64 – 75** | 12 | Underpromotions | 3 capture directions $\times$ 4 promotion pieces (Knight, Bishop, Rook, Queen) |

---

## 🌲 Two Ways to Think: Minimax vs MCTS

You can switch the engine's search brain dynamically in the Web Arena depending on how you want to challenge yourself:

```text
                           ┌───────────────────────────────┐
                           │      Neural Position Eval     │
                           │   Policy Head  │  Value Head  │
                           └───────────────┬───────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
         [ Alpha-Beta Minimax ]                       [ AlphaZero MCTS ]
    • Explores deep tactical lines               • Balances exploration & exploitation
    • Policy priors prune candidate branches     • Polynomial Upper Confidence (PUCT)
    • Zobrist transposition table caching        • Dirichlet root noise for variety
```

- **Policy-Guided $\alpha$-$\beta$ Minimax**: Fast, tactical, and sharp. It uses the neural policy head to rank moves first, producing rapid alpha-beta cutoffs.
- **AlphaZero Monte Carlo Tree Search (MCTS)**: Strategic, holistic, and creative. It runs hundreds of simulated rollouts guided by the **PUCT** formula:
  $$U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

---

## ⚡ Lightning Fast Inference

- **PyTorch GPU Acceleration**: Seamlessly runs on CUDA-enabled GPUs or multi-core CPUs.
- **ONNX Runtime INT8 Quantization**: Quantized from 106 MB down to **26 MB** for ultra-fast, lightweight inference without sacrificing tactical accuracy.

---

## 📜 Automated Match Gauntlets

Want to test the AI over a series of matches and export PGN game records?

```bash
# Run a 10-game self-play match at search depth 3
python benchmark.py --games 10 --depth 3
```

All game records are automatically formatted and saved to `pgn_exports/` so you can load and analyze them in Chess.com, Lichess, or ChessBase!

---

## 📄 License

Distributed under the open-source [MIT License](LICENSE). Happy playing, and good luck against the machine! ♟️