import pytest
import torch
import chess
from src.model import ChessResNet
from src.search import get_best_move as get_minimax_move
from src.mcts import get_best_move_mcts


def test_minimax_returns_legal_move():
    device = torch.device("cpu")
    model = ChessResNet(num_blocks=2, hidden_channels=32).to(device)
    model.eval()
    board = chess.Board()

    move = get_minimax_move(board, lookahead_depth=2, model=model, device=device)
    assert move is not None
    assert move in board.legal_moves


def test_mcts_returns_legal_move():
    device = torch.device("cpu")
    model = ChessResNet(num_blocks=2, hidden_channels=32).to(device)
    model.eval()
    board = chess.Board()

    move = get_best_move_mcts(board, model=model, device=device, num_simulations=20)
    assert move is not None
    assert move in board.legal_moves
