"""
Automated Benchmarking and Model Duel CLI for Chess-AI.
Supports running gauntlets against Stockfish or model self-play duels.
"""

import os
import sys
import argparse
import datetime
import uuid
from pathlib import Path
import torch
import chess
import chess.pgn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.model import ChessResNet
from src.onnx_engine import ONNXChessModel
from src.search import get_best_move as get_minimax_move
from src.mcts import get_best_move_mcts
from src.ui import BOLD, RESET, CYAN, GREEN, YELLOW, RED


def run_benchmark_duel(
    model_path: str = "models/chess_resnet_int8.onnx",
    num_games: int = 10,
    depth: int = 3,
    save_dir: str = "pgn_exports"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{CYAN}=== Starting Model Duel Benchmark ({num_games} Games) ==={RESET}")
    print(f"  Model: {model_path}")
    print(f"  Depth: {depth}")
    print(f"  Device: {device}\n")

    if model_path.endswith(".onnx") and os.path.exists(model_path):
        model = ONNXChessModel(model_path, use_gpu=(device.type == "cuda"))
    else:
        model = ChessResNet(num_blocks=10, hidden_channels=128).to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    wins_white = 0
    wins_black = 0
    draws = 0
    all_games = []

    for game_idx in range(num_games):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"Self-Play Benchmark Game {game_idx + 1}"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = f"ResNet (Depth {depth})"
        game.headers["Black"] = f"ResNet (Depth {depth})"
        node = game

        while not board.is_game_over() and not board.can_claim_draw():
            with torch.no_grad():
                move = get_minimax_move(board, lookahead_depth=depth, model=model, device=device)
            if move and move in board.legal_moves:
                board.push(move)
                node = node.add_variation(move)
            else:
                break

        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            draws += 1
            res_str = "1/2-1/2"
        elif outcome.winner == chess.WHITE:
            wins_white += 1
            res_str = "1-0"
        else:
            wins_black += 1
            res_str = "0-1"

        game.headers["Result"] = res_str
        all_games.append(game)
        print(f"  Game {game_idx + 1:02d}/{num_games:02d}: Result: {BOLD}{res_str}{RESET} (Moves: {len(board.move_stack)})")

    # Export combined PGN
    out_file = save_path / f"benchmark_match_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.pgn"
    with open(out_file, "w", encoding="utf-8") as f:
        for g in all_games:
            exporter = chess.pgn.FileExporter(f)
            g.accept(exporter)
            f.write("\n\n")

    print(f"\n{GREEN}✔ Benchmark Complete!{RESET}")
    print(f"  White Wins: {wins_white} | Black Wins: {wins_black} | Draws: {draws}")
    print(f"  Full PGN saved to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chess AI self-play benchmark")
    parser.add_argument("--model", type=str, default="models/chess_resnet_int8.onnx", help="Model weights path")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--depth", type=int, default=3, help="Search depth")
    args = parser.parse_args()

    run_benchmark_duel(model_path=args.model, num_games=args.games, depth=args.depth)
