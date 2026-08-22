"""
Chess-AI Web Application Server
Serves interactive web arena and REST API for neural network evaluation and move search.
"""

import os
import sys
import json
import time
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import chess
import torch

from src.model import ChessResNet
from src.search import get_best_move as get_minimax_move, get_model_evaluation
from src.mcts import get_best_move_mcts
from src.onnx_engine import ONNXChessModel, get_onnx_evaluation

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_TYPE = "pytorch"
MODEL_NAME = "chess_model_v3.pth"
GAME_BOARD = chess.Board()


def load_best_model():
    """Loads the pre-trained weights from models directory."""
    global MODEL, MODEL_TYPE, MODEL_NAME
    candidates = [
        ("models/chess_model_v3.pth", "pytorch"),
        ("models/chess_resnet_int8.onnx", "onnx"),
        ("models/chess_model.pth", "pytorch"),
        ("chess_model_v3.pth", "pytorch"),
        ("chess_resnet_int8.onnx", "onnx"),
    ]

    for path, mtype in candidates:
        if os.path.exists(path):
            MODEL_NAME = os.path.basename(path)
            MODEL_TYPE = mtype
            print(f"[*] Initializing {mtype.upper()} model: {path} on {DEVICE}...")
            if mtype == "onnx":
                MODEL = ONNXChessModel(path)
            else:
                MODEL = ChessResNet(num_blocks=10, hidden_channels=128).to(DEVICE)
                state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
                MODEL.load_state_dict(state_dict)
                MODEL.eval()
            return

    # Fallback to randomly initialized
    print("[!] No checkpoint found. Initializing demonstration weights...")
    MODEL = ChessResNet(num_blocks=10, hidden_channels=128).to(DEVICE)
    MODEL.eval()


def get_position_evaluation(board: chess.Board) -> float:
    """Evaluates position and returns score from White's perspective."""
    if MODEL_TYPE == "onnx":
        _, eval_score = get_onnx_evaluation(MODEL, board)
    else:
        _, eval_score = get_model_evaluation(MODEL, board, DEVICE)
    return float(eval_score)


class ChessAPIHandler(BaseHTTPRequestHandler):
    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_json_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self.serve_static_file("web/index.html", "text/html")
        elif path == "/static/style.css":
            self.serve_static_file("web/style.css", "text/css")
        elif path == "/static/app.js":
            self.serve_static_file("web/app.js", "application/javascript")
        elif path == "/api/status":
            self._set_json_headers()
            data = {
                "device": str(DEVICE),
                "model": MODEL_NAME,
                "modelType": MODEL_TYPE,
                "fen": GAME_BOARD.fen()
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body.decode("utf-8")) if post_body else {}

        if path == "/api/legal_moves":
            self.handle_legal_moves(data)
        elif path == "/api/apply_move":
            self.handle_apply_move(data)
        elif path == "/api/move":
            self.handle_ai_move(data)
        elif path == "/api/eval":
            self.handle_eval(data)
        elif path == "/api/undo":
            self.handle_undo(data)
        else:
            self.send_error(404, "Endpoint Not Found")

    def serve_static_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_legal_moves(self, data):
        fen = data.get("fen", GAME_BOARD.fen())
        square_str = data.get("square", "")
        board = chess.Board(fen)

        try:
            from_sq = chess.parse_square(square_str)
        except ValueError:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid square"}).encode("utf-8"))
            return

        moves = []
        for m in board.legal_moves:
            if m.from_square == from_sq:
                to_str = chess.square_name(m.to_square)
                is_capture = board.is_capture(m)
                is_promo = (m.promotion is not None)
                moves.append({
                    "to": to_str,
                    "uci": m.uci(),
                    "isCapture": is_capture,
                    "isPromotion": is_promo
                })

        self._set_json_headers()
        self.wfile.write(json.dumps({"moves": moves}).encode("utf-8"))

    def handle_apply_move(self, data):
        fen = data.get("fen", GAME_BOARD.fen())
        uci_str = data.get("uci", "")
        board = chess.Board(fen)

        try:
            move = chess.Move.from_uci(uci_str)
            if move not in board.legal_moves:
                self._set_json_headers(400)
                self.wfile.write(json.dumps({"error": "Illegal move"}).encode("utf-8"))
                return
        except ValueError:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid UCI format"}).encode("utf-8"))
            return

        is_capture = board.is_capture(move)
        san_str = board.san(move)
        board.push(move)
        GAME_BOARD.set_fen(board.fen())

        eval_score = get_position_evaluation(board)

        resp = {
            "fen": board.fen(),
            "san": san_str,
            "uci": uci_str,
            "isCapture": is_capture,
            "isCheck": board.is_check(),
            "isGameOver": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "eval": eval_score
        }

        self._set_json_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def handle_ai_move(self, data):
        fen = data.get("fen", GAME_BOARD.fen())
        engine_type = data.get("engine", "minimax")
        depth = int(data.get("depth", 3))
        simulations = int(data.get("simulations", 200))

        board = chess.Board(fen)
        if board.is_game_over():
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "Game is already over"}).encode("utf-8"))
            return

        start_time = time.perf_counter()

        if engine_type.lower() == "mcts":
            ai_move = get_best_move_mcts(board, model=MODEL, device=DEVICE, num_simulations=simulations)
        else:
            ai_move = get_minimax_move(board, lookahead_depth=depth, model=MODEL, device=DEVICE)

        elapsed = time.perf_counter() - start_time

        if ai_move is None or ai_move not in board.legal_moves:
            self._set_json_headers()
            self.wfile.write(json.dumps({"error": "AI resigned", "uci": None}).encode("utf-8"))
            return

        is_capture = board.is_capture(ai_move)
        san_str = board.san(ai_move)
        board.push(ai_move)
        GAME_BOARD.set_fen(board.fen())

        eval_score = get_position_evaluation(board)

        resp = {
            "fen": board.fen(),
            "san": san_str,
            "uci": ai_move.uci(),
            "isCapture": is_capture,
            "isCheck": board.is_check(),
            "isGameOver": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "eval": eval_score,
            "elapsed": elapsed
        }

        self._set_json_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def handle_eval(self, data):
        fen = data.get("fen", GAME_BOARD.fen())
        board = chess.Board(fen)
        eval_score = get_position_evaluation(board)
        self._set_json_headers()
        self.wfile.write(json.dumps({"eval": eval_score}).encode("utf-8"))

    def handle_undo(self, data):
        uci_moves = data.get("moves", None)
        board = chess.Board()

        if uci_moves is not None:
            for u in uci_moves:
                try:
                    m = chess.Move.from_uci(u)
                    if m in board.legal_moves:
                        board.push(m)
                except Exception:
                    pass
        else:
            steps = int(data.get("steps", 2))
            for _ in range(steps):
                if len(GAME_BOARD.move_stack) > 0:
                    GAME_BOARD.pop()
            board = GAME_BOARD.copy()

        eval_score = get_position_evaluation(board)
        last_m = None
        if len(board.move_stack) > 0:
            top_m = board.move_stack[-1]
            last_m = {
                "from": chess.square_name(top_m.from_square),
                "to": chess.square_name(top_m.to_square)
            }

        resp = {
            "fen": board.fen(),
            "lastMove": last_m,
            "eval": eval_score
        }
        self._set_json_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))


def run_server(port=8000, open_browser=True):
    load_best_model()
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChessAPIHandler)
    url = f"http://localhost:{port}"

    print(f"\n==============================================================")
    print(f"  ♟️  CHESS-AI WEB ARENA RUNNING AT: {url}")
    print(f"  Backend: {MODEL_TYPE.upper()} ({MODEL_NAME}) on {DEVICE}")
    print(f"==============================================================\n")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    run_server(port=port_arg, open_browser=True)
