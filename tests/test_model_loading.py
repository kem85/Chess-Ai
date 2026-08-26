import os
import pytest
import torch
import chess
from src.model import ChessResNet
from src.search import get_best_move, get_model_evaluation


def test_pytorch_resnet_weights_loading():
    checkpoint_path = "models/chess_model_v3.pth"
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 1_000_000:
        pytest.skip("Full PyTorch weights not present locally")

    device = torch.device("cpu")
    model = ChessResNet(num_blocks=10, hidden_channels=128).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    board = chess.Board()
    policy, value = get_model_evaluation(model, board, device)

    assert policy.shape == (1, 4864)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0

    move = get_best_move(board, lookahead_depth=2, model=model, device=device)
    assert move is not None
    assert move in board.legal_moves


def test_onnx_int8_loading():
    onnx_path = "models/chess_resnet_int8.onnx"
    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 100_000:
        pytest.skip("ONNX INT8 model not present locally")

    from src.onnx_engine import ONNXChessModel
    model = ONNXChessModel(onnx_path, use_gpu=False)
    device = torch.device("cpu")

    board = chess.Board()
    policy, value = get_model_evaluation(model, board, device)

    assert policy.shape == (1, 4864)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0

    move = get_best_move(board, lookahead_depth=2, model=model, device=device)
    assert move is not None
    assert move in board.legal_moves
