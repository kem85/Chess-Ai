import pytest
import numpy as np
import torch
import chess
from src.encoder import board_to_tensor, mirror_board_for_black, move_to_index


def test_board_to_tensor_shape_and_range():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == np.float32
    assert np.all((tensor == 0.0) | (tensor == 1.0))
    # 16 white pieces + 16 black pieces = 32 pieces active
    assert np.sum(tensor) == 32.0


def test_move_to_index_bounds():
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(move)
        assert idx is not None
        assert 0 <= idx < 4864


def test_mirror_board_for_black():
    board = chess.Board()
    t_white = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    t_mirrored = mirror_board_for_black(t_white)
    assert t_mirrored.shape == (1, 12, 8, 8)
