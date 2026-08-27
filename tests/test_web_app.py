import os
import json
import chess
import pytest
from app import WEB_DIR, BASE_DIR, get_model, get_position_evaluation, ChessAPIHandler


def test_web_static_assets_exist():
    """Verify all web assets required for the release package exist."""
    required_files = [
        "index.html",
        "style.css",
        "app.js",
        "chess.min.js"
    ]
    for fname in required_files:
        fpath = os.path.join(WEB_DIR, fname)
        assert os.path.exists(fpath), f"Missing web asset: {fpath}"
        assert os.path.getsize(fpath) > 0, f"Empty web asset: {fpath}"


def test_web_html_structure():
    """Check index.html contains essential arena elements."""
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "chessboard" in content
    assert "evalBarFill" in content
    assert "btnSoundToggle" in content
    assert "modelSelect" in content
    assert "engineSelect" in content


def test_model_resolution_and_eval():
    """Verify get_model resolves default ONNX model and computes evaluation."""
    model, model_type, name = get_model("onnx")
    assert model is not None
    assert model_type == "onnx"
    assert "onnx" in name.lower()

    board = chess.Board()
    eval_score = get_position_evaluation(board, model)
    assert isinstance(eval_score, float)
    assert -1.0 <= eval_score <= 1.0
