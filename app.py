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
from src.onnx_engine import ONNXChessModel
from src.search import get_best_move as get_minimax_move, get_model_evaluation
from src.mcts import get_best_move_mcts

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAME_BOARD = chess.Board()
SERVER_PORT = 8000
SERVER_THREAD = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Fallback in-memory cache for standalone HTTP server mode (dict survives for lifetime of process)
_STANDALONE_MODEL_CACHE: dict = {}


def ensure_model_file(target_path="models/chess_model_v3.pth") -> str:
    """Ensures full PyTorch weights are present, auto-downloading from GitHub LFS if needed."""
    resolved_target = target_path if os.path.isabs(target_path) else os.path.join(BASE_DIR, target_path)
    if os.path.exists(resolved_target) and os.path.getsize(resolved_target) > 1_000_000:
        return resolved_target
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000:
        return target_path

    url = "https://media.githubusercontent.com/media/kem85/Chess-Ai/main/models/chess_model_v3.pth"
    print(f"[*] Fetching full PyTorch ResNet weights from GitHub LFS ({url})...")
    os.makedirs(os.path.dirname(resolved_target), exist_ok=True)
    try:
        import urllib.request
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, resolved_target)
        if os.path.exists(resolved_target) and os.path.getsize(resolved_target) > 1_000_000:
            print(f"[✓] Full weights downloaded successfully ({os.path.getsize(resolved_target)} bytes).")
            return resolved_target
    except Exception as err:
        print(f"[!] Could not download checkpoint: {err}")
    return resolved_target


def _load_model_impl(backend: str):
    """
    Internal loader — actually initialises and returns (model, type, name).
    Called at most once per backend per process lifetime (cached by callers).
    """
    backend = (backend or "onnx").lower().strip()

    if backend == "onnx":
        onnx_candidates = [
            os.path.join(BASE_DIR, "models", "chess_resnet_int8.onnx"),
            "models/chess_resnet_int8.onnx",
            "chess_resnet_int8.onnx",
            os.path.join(BASE_DIR, "models", "chess_resnet.onnx"),
            "models/chess_resnet.onnx",
            "chess_resnet.onnx",
        ]
        for onnx_path in onnx_candidates:
            if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 100_000:
                try:
                    use_cuda = torch.cuda.is_available()
                    name = os.path.basename(onnx_path)
                    print(f"[*] Initializing ONNX Runtime Engine: {onnx_path} (GPU={use_cuda})...")
                    m = ONNXChessModel(onnx_path, use_gpu=use_cuda)
                    print(f"[✓] Successfully loaded ONNX engine from {onnx_path}")
                    return (m, "onnx", name)
                except Exception as err:
                    print(f"[!] Error loading ONNX model {onnx_path}: {err}")

    # PyTorch FP32 backend (.pth)
    ensure_model_file("models/chess_model_v3.pth")
    pytorch_candidates = [
        os.path.join(BASE_DIR, "models", "chess_model_v3.pth"),
        "models/chess_model_v3.pth",
        "models/chess_model.pth",
        os.path.join(BASE_DIR, "models", "chess_model.pth"),
        "chess_model_v3.pth",
        "chess_model.pth",
    ]
    for path in pytorch_candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            try:
                name = os.path.basename(path)
                print(f"[*] Initializing PyTorch ResNet model: {path} on {DEVICE}...")
                m = ChessResNet(num_blocks=10, hidden_channels=128).to(DEVICE)
                state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
                m.load_state_dict(state_dict)
                m.eval()
                print(f"[✓] Successfully loaded PyTorch weights from {path}")
                return (m, "pytorch", name)
            except Exception as err:
                print(f"[!] Error loading PyTorch {path}: {err}")

    # Fallback to demo weights
    print("[!] Checkpoint not found. Initializing demonstration weights...")
    m = ChessResNet(num_blocks=10, hidden_channels=128).to(DEVICE)
    m.eval()
    return (m, "pytorch", "demo_weights")


# ── Streamlit-cached loader (survives st.rerun() / script re-execution) ────────
# We define the @st.cache_resource version only when Streamlit is actually
# available, so standalone python app.py still works with the plain dict below.
try:
    import streamlit as _st_probe

    @_st_probe.cache_resource(show_spinner=False)
    def _st_cached_model(backend: str):
        """Streamlit resource cache — loaded ONCE, shared across all reruns."""
        return _load_model_impl(backend)

    def get_model(backend: str = "onnx"):
        """Returns (model, type, name) — cached via st.cache_resource in Streamlit mode."""
        return _st_cached_model((backend or "onnx").lower().strip())

except Exception:
    # Standalone / non-Streamlit fallback: plain in-process dict cache
    def get_model(backend: str = "onnx"):  # type: ignore[misc]
        """Returns (model, type, name) — cached in-process dict for standalone HTTP mode."""
        key = (backend or "onnx").lower().strip()
        if key not in _STANDALONE_MODEL_CACHE:
            _STANDALONE_MODEL_CACHE[key] = _load_model_impl(key)
        return _STANDALONE_MODEL_CACHE[key]


def load_best_model():
    """Pre-warms the default ONNX model into memory."""
    get_model("onnx")


def get_position_evaluation(board: chess.Board, model=None) -> float:
    """Evaluates position and returns score from White's perspective."""
    if model is None:
        model, _, _ = get_model("onnx")
    _, eval_score = get_model_evaluation(model, board, DEVICE)
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
            self.serve_static_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        elif path in ["/static/style.css", "/style.css"]:
            self.serve_static_file(os.path.join(WEB_DIR, "style.css"), "text/css")
        elif path in ["/static/chess.min.js", "/chess.min.js"]:
            self.serve_static_file(os.path.join(WEB_DIR, "chess.min.js"), "application/javascript")
        elif path in ["/static/app.js", "/app.js"]:
            self.serve_static_file(os.path.join(WEB_DIR, "app.js"), "application/javascript")
        elif path == "/api/status":
            self._set_json_headers()
            active_m, active_type, active_name = get_model("onnx")
            data = {
                "device": str(DEVICE),
                "model": active_name,
                "modelType": active_type,
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
        model_backend = data.get("modelBackend", "onnx")
        model, _, _ = get_model(model_backend)
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

        eval_score = get_position_evaluation(board, model)

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
        model_backend = data.get("modelBackend", "onnx")
        depth = int(data.get("depth", 3))
        simulations = int(data.get("simulations", 200))
        board = chess.Board(fen)

        if board.is_game_over():
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "Game is over"}).encode("utf-8"))
            return

        model, model_type, model_name = get_model(model_backend)

        start_t = time.time()
        if engine_type == "mcts":
            best_move = get_best_move_mcts(board, model, DEVICE, num_simulations=simulations)
        else:
            best_move = get_minimax_move(board, depth, model, DEVICE)

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

        eval_score = get_position_evaluation(board, model)

        resp = {
            "uci": uci_str,
            "san": san_str,
            "fen": board.fen(),
            "isCapture": is_capture,
            "isCheck": board.is_check(),
            "isGameOver": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "eval": eval_score,
            "calcTimeMs": round(calc_time_ms, 1),
            "model": model_name,
            "modelType": model_type
        }

        self._set_json_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def handle_eval(self, data):
        fen = data.get("fen", GAME_BOARD.fen())
        model_backend = data.get("modelBackend", "onnx")
        model, _, _ = get_model(model_backend)
        board = chess.Board(fen)
        eval_score = get_position_evaluation(board, model)
        self._set_json_headers()
        self.wfile.write(json.dumps({"eval": eval_score}).encode("utf-8"))

    def handle_undo(self, data):
        count = int(data.get("count", 1))
        model_backend = data.get("modelBackend", "onnx")
        model, _, _ = get_model(model_backend)
        for _ in range(count):
            if len(GAME_BOARD.move_stack) > 0:
                GAME_BOARD.pop()

        eval_score = get_position_evaluation(GAME_BOARD, model)
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

    active_m, active_type, active_name = get_model("onnx")
    component_value = chess_arena_component(
        ai_result=st.session_state["ai_result"],
        device=str(DEVICE),
        model=active_name,
        model_type=active_type,
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
            model_backend = component_value.get("modelBackend", "onnx")
            depth = int(component_value.get("depth", 3))
            simulations = int(component_value.get("simulations", 200))

            board = chess.Board(fen)
            if not board.is_game_over():
                model, model_type, model_name = get_model(model_backend)
                start_t = time.time()
                if engine_type == "mcts":
                    best_move = get_best_move_mcts(board, model, DEVICE, num_simulations=simulations)
                else:
                    best_move = get_minimax_move(board, depth, model, DEVICE)
                calc_time_ms = (time.time() - start_t) * 1000.0

                if best_move is not None:
                    is_capture = board.is_capture(best_move)
                    san_str = board.san(best_move)
                    uci_str = best_move.uci()
                    board.push(best_move)
                    eval_score = get_position_evaluation(board, model)

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
                        "calcTimeMs": round(calc_time_ms, 1),
                        "model": model_name,
                        "modelType": model_type
                    }
                    st.rerun()


def main(open_browser: bool = True):
    """Main CLI entrypoint for launching the standalone Chess-AI Web Arena."""
    load_best_model()
    port = find_available_port(8000)
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChessAPIHandler)
    url = f"http://127.0.0.1:{port}"

    active_m, active_type, active_name = get_model("onnx")
    print(f"\n==============================================================")
    print(f"⚡ Chess-AI Web Arena running on: {url}")
    print(f"🧠 Engine Device: {DEVICE}")
    print(f"📦 Default Model: {active_name} ({active_type.upper()})")
    print(f"🎯 Available:     ONNX INT8 (Fast) | PyTorch FP32 (High Precision)")
    print(f"==============================================================\n")

    if open_browser:
        threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Chess-AI Server...")
        httpd.server_close()


if not is_st_active and __name__ == "__main__":
    main()
