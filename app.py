"""
Chess-AI Web Application Server & Streamlit Edition
Serves interactive web arena with on-board piece movement, vertical evaluation bar,
and REST API for neural network evaluation and move search.
Hostable on Streamlit Cloud or standalone via python app.py.
"""

import os
import sys
import json
import time
import socket
import webbrowser
import threading

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import chess
import torch

from src.model import ChessResNet
from src.search import get_best_move as get_minimax_move, get_model_evaluation
from src.mcts import get_best_move_mcts

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_TYPE = "pytorch"
MODEL_NAME = "chess_model_v3.pth"
GAME_BOARD = chess.Board()
SERVER_PORT = 8000
SERVER_THREAD = None


def ensure_model_file(target_path="models/chess_model_v3.pth") -> str:
    """Ensures full PyTorch weights are present, auto-downloading from GitHub LFS if needed."""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000:
        return target_path

    url = "https://media.githubusercontent.com/media/kem85/Chess-Ai/main/models/chess_model_v3.pth"
    print(f"[*] Fetching full PyTorch ResNet weights from GitHub LFS ({url})...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        import urllib.request
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, target_path)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000:
            print(f"[✓] Full weights downloaded successfully ({os.path.getsize(target_path)} bytes).")
    except Exception as err:
        print(f"[!] Could not download checkpoint: {err}")
    return target_path


def load_best_model():
    """Loads the pre-trained PyTorch ResNet weights from models directory."""
    global MODEL, MODEL_TYPE, MODEL_NAME
    if MODEL is not None:
        return

    ensure_model_file("models/chess_model_v3.pth")

    candidates = [
        "models/chess_model_v3.pth",
        "models/chess_model.pth",
        "chess_model_v3.pth",
        "chess_model.pth",
    ]

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            try:
                MODEL_NAME = os.path.basename(path)
                MODEL_TYPE = "pytorch"
                print(f"[*] Initializing PyTorch ResNet model: {path} on {DEVICE}...")
                MODEL = ChessResNet(num_blocks=10, hidden_channels=128).to(DEVICE)
                state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
                MODEL.load_state_dict(state_dict)
                MODEL.eval()
                print(f"[✓] Successfully loaded weights from {path}")
                return
            except Exception as err:
                print(f"[!] Error loading {path}: {err}")

    # Fallback to randomly initialized ResNet if no valid weights found
    print("[!] No valid checkpoint (>1MB) found. Initializing demonstration weights...")
    MODEL = ChessResNet(num_blocks=10, hidden_channels=128).to(DEVICE)
    MODEL.eval()


def get_position_evaluation(board: chess.Board) -> float:
    """Evaluates position and returns score from White's perspective."""
    _, eval_score = get_model_evaluation(MODEL, board, DEVICE)
    return float(eval_score)


class ChessAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass


    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Access-Control-Request-Private-Network")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_json_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self.serve_static_file("web/index.html", "text/html")
        elif path in ["/static/style.css", "/style.css"]:
            self.serve_static_file("web/style.css", "text/css")
        elif path in ["/static/chess.min.js", "/chess.min.js"]:
            self.serve_static_file("web/chess.min.js", "application/javascript")
        elif path in ["/static/app.js", "/app.js"]:
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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
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
            self.wfile.write(json.dumps({"error": "Game is over"}).encode("utf-8"))
            return

        start_t = time.time()
        if engine_type == "mcts":
            best_move = get_best_move_mcts(board, MODEL, DEVICE, num_simulations=simulations)
        else:
            best_move = get_minimax_move(board, depth, MODEL, DEVICE)

        calc_time_ms = (time.time() - start_t) * 1000.0

        if best_move is None:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "No move found"}).encode("utf-8"))
            return

        is_capture = board.is_capture(best_move)
        san_str = board.san(best_move)
        uci_str = best_move.uci()
        board.push(best_move)
        GAME_BOARD.set_fen(board.fen())

        eval_score = get_position_evaluation(board)

        resp = {
            "uci": uci_str,
            "san": san_str,
            "fen": board.fen(),
            "isCapture": is_capture,
            "isCheck": board.is_check(),
            "isGameOver": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "eval": eval_score,
            "calcTimeMs": round(calc_time_ms, 1)
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
        count = int(data.get("count", 1))
        for _ in range(count):
            if len(GAME_BOARD.move_stack) > 0:
                GAME_BOARD.pop()

        eval_score = get_position_evaluation(GAME_BOARD)
        self._set_json_headers()
        self.wfile.write(json.dumps({
            "fen": GAME_BOARD.fen(),
            "eval": eval_score
        }).encode("utf-8"))


def find_available_port(start_port=8000):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def start_backend_server(port=8000):
    global SERVER_PORT, SERVER_THREAD
    if SERVER_THREAD is not None and SERVER_THREAD.is_alive():
        return SERVER_PORT

    load_best_model()
    SERVER_PORT = find_available_port(port)
    server_address = ("0.0.0.0", SERVER_PORT)
    httpd = HTTPServer(server_address, ChessAPIHandler)

    def serve():
        httpd.serve_forever()

    SERVER_THREAD = threading.Thread(target=serve, daemon=True)
    SERVER_THREAD.start()
    return SERVER_PORT


# Detect if running under Streamlit context
try:
    import streamlit as st
    IS_STREAMLIT = True
except ImportError:
    IS_STREAMLIT = False

if IS_STREAMLIT:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_st_active = get_script_run_ctx() is not None
    except Exception:
        is_st_active = True
else:
    is_st_active = False

if is_st_active:
    # --- STREAMLIT HOSTING MODE ---
    load_best_model()

    st.set_page_config(
        page_title="Chess-AI — Universal Neural Engine & Arena",
        page_icon="♟️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
        <style>
            html, body, [data-testid="stAppViewContainer"], [data-testid="stMainViewContainer"], [data-testid="stMain"], section.main {
                overflow-x: hidden !important;
                overflow-y: auto !important;
                padding: 0 !important;
                margin: 0 !important;
                height: auto !important;
                min-height: 100vh !important;
                background-color: #07090e !important;
            }
            .block-container {
                padding: 0rem !important;
                margin: 0 !important;
                max-width: 100% !important;
            }
            header[data-testid="stHeader"], footer, #MainMenu {
                display: none !important;
            }
            iframe {
                border: none !important;
                width: 100% !important;
                min-height: 100vh !important;
            }
        </style>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "web"))
    chess_arena_component = components.declare_component("chess_arena", path=web_dir)

    if "ai_result" not in st.session_state:
        st.session_state["ai_result"] = None
    if "last_action_id" not in st.session_state:
        st.session_state["last_action_id"] = None

    component_value = chess_arena_component(
        ai_result=st.session_state["ai_result"],
        device=str(DEVICE),
        model=MODEL_NAME,
        model_type=MODEL_TYPE,
        default=None,
        key="chess_arena_comp"
    )

    if component_value and isinstance(component_value, dict):
        action = component_value.get("action")
        action_id = component_value.get("actionId")
        if action == "ai_move" and action_id != st.session_state["last_action_id"]:
            st.session_state["last_action_id"] = action_id
            fen = component_value.get("fen", GAME_BOARD.fen())
            engine_type = component_value.get("engine", "minimax")
            depth = int(component_value.get("depth", 3))
            simulations = int(component_value.get("simulations", 200))

            board = chess.Board(fen)
            if not board.is_game_over():
                start_t = time.time()
                if engine_type == "mcts":
                    best_move = get_best_move_mcts(board, MODEL, DEVICE, num_simulations=simulations)
                else:
                    best_move = get_minimax_move(board, depth, MODEL, DEVICE)
                calc_time_ms = (time.time() - start_t) * 1000.0

                if best_move is not None:
                    is_capture = board.is_capture(best_move)
                    san_str = board.san(best_move)
                    uci_str = best_move.uci()
                    board.push(best_move)
                    eval_score = get_position_evaluation(board)

                    st.session_state["ai_result"] = {
                        "actionId": action_id,
                        "uci": uci_str,
                        "san": san_str,
                        "fen": board.fen(),
                        "isCapture": is_capture,
                        "isCheck": board.is_check(),
                        "isGameOver": board.is_game_over(),
                        "result": board.result() if board.is_game_over() else None,
                        "eval": eval_score,
                        "calcTimeMs": round(calc_time_ms, 1)
                    }
                    st.rerun()


elif __name__ == "__main__":
    # --- STANDALONE PYTHON MODE ---
    load_best_model()
    port = find_available_port(8000)
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChessAPIHandler)
    url = f"http://127.0.0.1:{port}"

    print(f"\n==============================================================")
    print(f"⚡ Chess-AI Web Arena running on: {url}")
    print(f"🧠 Engine Device: {DEVICE}")
    print(f"📦 Active Model:  {MODEL_NAME} ({MODEL_TYPE.upper()})")
    print(f"==============================================================\n")

    threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Chess-AI Server...")
        httpd.server_close()
