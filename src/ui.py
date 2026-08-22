"""
Rich ANSI Terminal UI for Chess-AI.
Provides styled board rendering, dynamic evaluation bar, piece counter, and search telemetry.
"""

from typing import Optional, List
import sys
import chess

# ANSI Color & Style Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Colors
CYAN = "\033[36m"
BRIGHT_CYAN = "\033[96m"
GREEN = "\033[32m"
BRIGHT_GREEN = "\033[92m"
YELLOW = "\033[33m"
BRIGHT_YELLOW = "\033[93m"
RED = "\033[31m"
BRIGHT_RED = "\033[91m"
WHITE = "\033[97m"
GRAY = "\033[90m"
BG_DARK = "\033[48;5;234m"
BG_LIGHT = "\033[48;5;238m"

# Unicode piece mapping
UNICODE_PIECES = {
    'r': f"{BRIGHT_CYAN}♜{RESET}",
    'n': f"{BRIGHT_CYAN}♞{RESET}",
    'b': f"{BRIGHT_CYAN}♝{RESET}",
    'q': f"{BRIGHT_CYAN}♛{RESET}",
    'k': f"{BRIGHT_CYAN}♚{RESET}",
    'p': f"{BRIGHT_CYAN}♟{RESET}",
    'R': f"{BRIGHT_YELLOW}♖{RESET}",
    'N': f"{BRIGHT_YELLOW}♘{RESET}",
    'B': f"{BRIGHT_YELLOW}♗{RESET}",
    'Q': f"{BRIGHT_YELLOW}♕{RESET}",
    'K': f"{BRIGHT_YELLOW}♔{RESET}",
    'P': f"{BRIGHT_YELLOW}♙{RESET}",
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


def format_eval_bar(eval_score: float, width: int = 24) -> str:
    """
    Renders an ASCII/Unicode dynamic evaluation bar:
    [-1.0 (Black Win) <-------- [===|===] --------> +1.0 (White Win)]
    """
    # Clamp score to [-1.0, 1.0]
    clamped = max(-1.0, min(1.0, eval_score))
    # Normalized position [0.0, 1.0]
    norm = (clamped + 1.0) / 2.0
    pos = int(round(norm * width))
    pos = max(0, min(width, pos))

    bar = ""
    for i in range(width + 1):
        if i == width // 2:
            bar += f"{GRAY}│{RESET}"
        elif i < pos:
            bar += f"{BRIGHT_GREEN}█{RESET}"
        else:
            bar += f"{GRAY}░{RESET}"

    sign = "+" if eval_score > 0 else ""
    return f"[{bar}] {BOLD}{sign}{eval_score:+.2f}{RESET}"


def get_captured_pieces(board: chess.Board) -> str:
    """Calculates captured pieces and material imbalance."""
    starting_counts = {
        chess.WHITE: {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1},
        chess.BLACK: {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
    }
    
    current_counts = {
        chess.WHITE: {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0, chess.ROOK: 0, chess.QUEEN: 0},
        chess.BLACK: {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0, chess.ROOK: 0, chess.QUEEN: 0}
    }

    white_mat = 0
    black_mat = 0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            current_counts[piece.color][piece.piece_type] += 1
            if piece.color == chess.WHITE:
                white_mat += PIECE_VALUES[piece.piece_type]
            else:
                black_mat += PIECE_VALUES[piece.piece_type]

    # White captures (Black pieces lost)
    white_caps = []
    for ptype, initial in starting_counts[chess.BLACK].items():
        lost = initial - current_counts[chess.BLACK][ptype]
        if lost > 0:
            sym = UNICODE_PIECES[chess.Piece(ptype, chess.BLACK).symbol()]
            white_caps.append(f"{sym}{f'x{lost}' if lost > 1 else ''}")

    # Black captures (White pieces lost)
    black_caps = []
    for ptype, initial in starting_counts[chess.WHITE].items():
        lost = initial - current_counts[chess.WHITE][ptype]
        if lost > 0:
            sym = UNICODE_PIECES[chess.Piece(ptype, chess.WHITE).symbol()]
            black_caps.append(f"{sym}{f'x{lost}' if lost > 1 else ''}")

    diff = white_mat - black_mat
    diff_str = f" ({'+' if diff > 0 else ''}{diff})" if diff != 0 else ""

    w_str = " ".join(white_caps) if white_caps else f"{DIM}None{RESET}"
    b_str = " ".join(black_caps) if black_caps else f"{DIM}None{RESET}"

    return f"  {BRIGHT_YELLOW}White Captures:{RESET} {w_str} {BOLD}{diff_str if diff > 0 else ''}{RESET}\n  {BRIGHT_CYAN}Black Captures:{RESET} {b_str} {BOLD}{diff_str if diff < 0 else ''}{RESET}"


def render_board(
    board: chess.Board,
    last_move: Optional[chess.Move] = None,
    eval_score: Optional[float] = None
) -> str:
    """
    Renders a terminal chess board with highlighted move squares and coordinates.
    """
    from_sq = last_move.from_square if last_move else None
    to_sq = last_move.to_square if last_move else None

    lines = []
    lines.append(f"\n   {GRAY}┌───┬───┬───┬───┬───┬───┬───┬───┐{RESET}")

    for r in range(7, -1, -1):
        row = [f" {BOLD}{r + 1}{RESET} {GRAY}│{RESET}"]
        for f in range(8):
            sq = chess.square(f, r)
            piece = board.piece_at(sq)

            # Highlight last move squares
            is_highlight = (sq == from_sq or sq == to_sq)
            
            if piece:
                sym = UNICODE_PIECES.get(piece.symbol(), piece.symbol())
            else:
                sym = f"{GRAY}·{RESET}"

            if is_highlight:
                row.append(f" {BG_LIGHT}{sym}{RESET} {GRAY}│{RESET}")
            else:
                row.append(f" {sym} {GRAY}│{RESET}")

        lines.append("".join(row))
        if r > 0:
            lines.append(f"   {GRAY}├───┼───┼───┼───┼───┼───┼───┼───┤{RESET}")

    lines.append(f"   {GRAY}└───┴───┴───┴───┴───┴───┴───┴───┘{RESET}")
    lines.append(f"     {BOLD}a   b   c   d   e   f   g   h{RESET}\n")

    if eval_score is not None:
        lines.append(f"  {BOLD}Position Eval:{RESET} {format_eval_bar(eval_score)}")

    lines.append(get_captured_pieces(board))
    return "\n".join(lines)
