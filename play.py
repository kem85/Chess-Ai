"""
Interactive CLI for Chess-AI: Play against Deep Residual Neural Network with Minimax or MCTS.
"""

import os
import sys
import argparse
import time
from typing import Optional, Tuple
import torch
import chess

from src.model import ChessResNet
from src.onnx_engine import ONNXChessModel
from src.search import get_best_move as get_best_move_minimax, get_model_evaluation
from src.mcts import get_best_move_mcts
from src.ui import render_board, BOLD, RESET, CYAN, YELLOW, GREEN, RED, DIM

# Fix Windows console encoding for UTF-8 box-drawing & unicode pieces
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def resolve_default_model(model_arg: str = None) -> str:
    """Finds the best available pre-trained checkpoint or resolves aliases ('onnx'/'pytorch')."""
    if model_arg:
        if model_arg.lower() in ("onnx", "int8"):
            for p in ["models/chess_resnet_int8.onnx", "chess_resnet_int8.onnx"]:
                if os.path.exists(p): return p
        elif model_arg.lower() in ("pytorch", "pth", "fp32"):
            for p in ["models/chess_model_v3.pth", "models/chess_model.pth", "chess_model_v3.pth"]:
                if os.path.exists(p): return p
        return model_arg

    candidates = [
        "models/chess_resnet_int8.onnx",
        "chess_resnet_int8.onnx",
        "models/chess_resnet.onnx",
        "chess_resnet.onnx",
        "models/chess_model_v3.pth",
        "models/chess_model.pth",
        "chess_model_v3.pth",
        "chess_model.pth"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 100_000:
            return c
    return "models/chess_resnet_int8.onnx"


def load_model(model_path: str, device: torch.device) -> Tuple[object, str]:
    """Loads ONNX Runtime model or PyTorch ResNet model from checkpoint with telemetry."""
    model_path = resolve_default_model(model_path)
    if model_path.endswith(".onnx") and os.path.exists(model_path):
        use_gpu = (device.type == "cuda")
        print(f"[*] Loading ONNX Runtime Engine from: {model_path} (GPU={use_gpu})")
        print(f"    ⚡ ONNX INT8: 8-bit quantized weights, 75% smaller footprint, fast CPU inference.")
        model = ONNXChessModel(model_path, use_gpu=use_gpu)
        print(f"[✓] ONNX engine loaded successfully.")
        return model, "onnx"

    model = ChessResNet(num_blocks=10, hidden_channels=128).to(device)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
        try:
            print(f"[*] Loading PyTorch Full Precision (FP32) weights from: {model_path}")
            print(f"    🎯 PyTorch FP32: Full 32-bit floating-point weights (highest raw precision, slower CPU inference).")
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[✓] PyTorch model loaded successfully.")
        except Exception as err:
            print(f"[!] Failed to load {model_path}: {err}")
    else:
        print(f"[!] Warning: Model checkpoint '{model_path}' not found or invalid size.")
        print("[!] Running with randomly initialized weights for demonstration.\n")
    model.eval()
    return model, "pytorch"


def print_banner(model_name: str, engine_name: str, color_name: str, depth_or_sims: str):
    print(f"\n{CYAN}=============================================================={RESET}")
    print(f"{BOLD}                ♟️  CHESS-AI: MAN VS MACHINE  ♟️                {RESET}")
    print(f"{CYAN}=============================================================={RESET}")
    print(f"  {BOLD}Model Checkpoint:{RESET} {model_name}")
    print(f"  {BOLD}Engine Algorithm:{RESET} {engine_name.upper()} ({depth_or_sims})")
    print(f"  {BOLD}Human Player:{RESET}     {color_name.upper()}")
    print(f"  {BOLD}Commands:{RESET}         Enter move in SAN (e.g. {GREEN}e4{RESET}, {GREEN}Nf6{RESET}, {GREEN}O-O{RESET}) or UCI ({GREEN}e2e4{RESET})")
    print(f"                    Type {RED}'undo'{RESET} to take back, or {RED}'quit'{RESET} to resign.")
    print(f"{CYAN}=============================================================={RESET}\n")


def play_game(
    model_path: str,
    engine: str = "minimax",
    depth: int = 3,
    simulations: int = 200,
    play_as: str = "black"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_model(model_path, device)

    board = chess.Board()
    user_is_black = (play_as.lower() in ["b", "black"])
    last_move = None
    eval_score = 0.0

    depth_or_sims = f"{simulations} simulations" if engine.lower() == "mcts" else f"Depth {depth}"
    print_banner(
        model_name=os.path.basename(model_path),
        engine_name=engine,
        color_name="Black" if user_is_black else "White",
        depth_or_sims=depth_or_sims
    )

    move_history = []

    while not board.is_game_over():
        print(render_board(board, last_move=last_move, eval_score=eval_score))

        is_ai_turn = (board.turn == chess.WHITE and user_is_black) or (board.turn == chess.BLACK and not user_is_black)

        if is_ai_turn:
            ai_color_name = "White" if board.turn == chess.WHITE else "Black"
            print(f"\n🤖 {BOLD}AI ({ai_color_name}) is calculating...{RESET}")
            start_t = time.perf_counter()

            if engine.lower() == "mcts":
                ai_move = get_best_move_mcts(board, model=model, device=device, num_simulations=simulations)
            else:
                ai_move = get_best_move_minimax(board, lookahead_depth=depth, model=model, device=device)

            elapsed = time.perf_counter() - start_t

            if ai_move is None or ai_move not in board.legal_moves:
                print(f"{RED}AI resigns! Congratulations!{RESET}")
                break

            san_str = board.san(ai_move)
            board.push(ai_move)
            last_move = ai_move
            move_history.append(san_str)

            _, eval_score = get_model_evaluation(model, board, device)

            print(f"🤖 {BOLD}AI played:{RESET} {GREEN}{san_str}{RESET} {DIM}(uci: {ai_move.uci()}, {elapsed:.2f}s){RESET}\n")

        else:
            human_color_name = "Black" if user_is_black else "White"
            while True:
                user_input = input(f"\n👉 {BOLD}Your Move ({human_color_name}):{RESET} ").strip()
                
                if user_input.lower() in ["quit", "exit", "resign"]:
                    print(f"\n{RED}Game resigned by player. AI wins by default.{RESET}")
                    return

                if user_input.lower() in ["undo", "takeback", "u"]:
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        last_move = board.move_stack[-1] if board.move_stack else None
                        print(f"{YELLOW}[*] Undid last move pair.{RESET}")
                        break
                    else:
                        print(f"{YELLOW}[!] Nothing to undo.{RESET}")
                        continue

                # Try SAN (e.g. e4, Nf3, exd5, O-O)
                try:
                    move = board.parse_san(user_input)
                    san_str = board.san(move)
                    board.push(move)
                    last_move = move
                    move_history.append(san_str)
                    break
                except ValueError:
                    pass

                # Try UCI (e.g. e2e4)
                try:
                    move = chess.Move.from_uci(user_input.lower())
                    if move in board.legal_moves:
                        san_str = board.san(move)
                        board.push(move)
                        last_move = move
                        move_history.append(san_str)
                        break
                    else:
                        print(f"{RED}❌ Illegal move. Legal options include: {', '.join([board.san(m) for m in list(board.legal_moves)[:6]])}...{RESET}")
                except ValueError:
                    print(f"{RED}❌ Unrecognized notation. Use standard SAN (e.g. e4, Nf6, exd5) or UCI (e2e4).{RESET}")

    print(render_board(board, last_move=last_move, eval_score=eval_score))
    print(f"\n{CYAN}=============================================================={RESET}")
    print(f"{BOLD}                        GAME OVER                             {RESET}")
    print(f"{CYAN}=============================================================={RESET}")
    print(f"  {BOLD}Final Result:{RESET} {board.result()}")
    
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"  {GREEN}🏆 Checkmate! Winner: {winner}{RESET}")
    elif board.is_stalemate():
        print(f"  {YELLOW}🤝 Draw by Stalemate.{RESET}")
    elif board.is_insufficient_material():
        print(f"  {YELLOW}🤝 Draw by Insufficient Material.{RESET}")
    elif board.can_claim_threefold_repetition():
        print(f"  {YELLOW}🤝 Draw by Threefold Repetition.{RESET}")
    elif board.can_claim_fifty_moves():
        print(f"  {YELLOW}🤝 Draw by Fifty-Move Rule.{RESET}")
    print(f"{CYAN}=============================================================={RESET}\n")


if __name__ == "__main__":
    default_model = resolve_default_model()
    parser = argparse.ArgumentParser(description="Play against the Dual-Head Chess AI (ONNX INT8 or PyTorch FP32)")
    parser.add_argument("--model", type=str, default=default_model, help="Model weights path or alias: 'onnx' (fast INT8, 26MB) or 'pytorch' (high precision FP32 .pth, slower)")
    parser.add_argument("--engine", type=str, default="minimax", choices=["minimax", "mcts"], help="Search algorithm (minimax / mcts)")
    parser.add_argument("--depth", type=int, default=3, help="Minimax search depth (default: 3)")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulation count (default: 200)")
    parser.add_argument("--color", type=str, default="black", choices=["white", "black"], help="Play as 'white' or 'black'")
    args = parser.parse_args()

    play_game(
        model_path=args.model,
        engine=args.engine,
        depth=args.depth,
        simulations=args.simulations,
        play_as=args.color
    )
