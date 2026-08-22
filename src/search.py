from typing import Tuple, List, Optional, Any
import numpy as np
import torch
import chess
from .encoder import board_to_tensor, move_to_index

# Global transposition caches
evaluation_cache = {}
search_cache = {}

eval_cache_hits = 0
eval_cache_misses = 0
search_cache_hits = 0
search_cache_misses = 0


def get_model_evaluation(
    model: Any,
    board: chess.Board,
    device: torch.device
) -> Tuple[torch.Tensor, float]:
    """
    Evaluates board position with either PyTorch or ONNX model.
    Returns (policy_logits, value_score_from_white_perspective).
    """
    global eval_cache_hits, eval_cache_misses

    model_id = id(model)
    board_key = board._transposition_key()
    cache_key = (model_id, board_key)

    if cache_key in evaluation_cache:
        eval_cache_hits += 1
        return evaluation_cache[cache_key]

    eval_cache_misses += 1

    is_black = (board.turn == chess.BLACK)
    eval_board = board.mirror() if is_black else board
    np_board = board_to_tensor(eval_board)

    # Check if ONNX model
    if hasattr(model, "session"):
        input_array = np.expand_dims(np_board, axis=0)
        p_out, v_out = model(input_array)
        policy_out = torch.from_numpy(p_out)
        val_item = float(v_out.item())
    else:
        input_tensor = torch.from_numpy(np_board).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode():
            policy_out, value_out = model(input_tensor)
        val_item = float(value_out.item())

    final_eval = val_item * (-1.0 if is_black else 1.0)
    result = (policy_out, final_eval)
    evaluation_cache[cache_key] = result

    if len(evaluation_cache) > 1_000_000:
        evaluation_cache.clear()

    return result


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    model: Any,
    device: torch.device,
    top_candidates: int = 8
) -> Tuple[float, List[chess.Move]]:
    """
    Alpha-Beta Minimax search guided by Policy priors and NN position evaluations.
    """
    global search_cache_hits, search_cache_misses

    cache_key = (board._transposition_key(), depth, is_maximizing)

    if cache_key in search_cache:
        search_cache_hits += 1
        return search_cache[cache_key]

    search_cache_misses += 1

    if board.is_checkmate():
        score = (-10000.0 - depth) if is_maximizing else (10000.0 + depth)
        result = (score, [])
        search_cache[cache_key] = result
        return result

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
        or board.is_repetition(3)
    ):
        result = (0.0, [])
        search_cache[cache_key] = result
        return result

    policy_logits, value_score = get_model_evaluation(model, board, device)

    if depth <= 0:
        result = (value_score, [])
        search_cache[cache_key] = result
        return result

    legal_moves = list(board.legal_moves)
    is_black = (board.turn == chess.BLACK)
    policy_scores = policy_logits.squeeze(0)

    moves_with_scores = []
    for move in legal_moves:
        if is_black:
            mirrored_move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                move.promotion
            )
            move_idx = move_to_index(mirrored_move)
        else:
            move_idx = move_to_index(move)

        if move_idx is None:
            continue

        score = policy_scores[move_idx].item()
        moves_with_scores.append((score, move))

    moves_with_scores.sort(key=lambda x: x[0], reverse=True)
    moves_with_scores = moves_with_scores[:top_candidates]

    best_path = []

    if is_maximizing:
        best_eval = float("-inf")
        for _, move in moves_with_scores:
            board.push(move)
            eval_score, path = minimax(board, depth - 1, alpha, beta, False, model, device, top_candidates)
            board.pop()

            if eval_score > best_eval:
                best_eval = eval_score
                best_path = [move] + path

            alpha = max(alpha, best_eval)
            if beta <= alpha:
                break

        result = (best_eval, best_path)
        search_cache[cache_key] = result
        return result
    else:
        best_eval = float("inf")
        for _, move in moves_with_scores:
            board.push(move)
            eval_score, path = minimax(board, depth - 1, alpha, beta, True, model, device, top_candidates)
            board.pop()

            if eval_score < best_eval:
                best_eval = eval_score
                best_path = [move] + path

            beta = min(beta, best_eval)
            if beta <= alpha:
                break

        result = (best_eval, best_path)
        search_cache[cache_key] = result
        return result


def get_best_move(
    board_state: chess.Board,
    lookahead_depth: int,
    model: Any,
    device: torch.device,
    candidate_pruning: int = 12
) -> Optional[chess.Move]:
    """
    Finds the optimal move using Policy-guided root exploration and alpha-beta minimax.
    """
    global search_cache, search_cache_hits, search_cache_misses

    search_cache.clear()
    search_cache_hits = 0
    search_cache_misses = 0

    is_white_turn = (board_state.turn == chess.WHITE)
    best_move = None

    policy_out, _ = get_model_evaluation(model, board_state, device)
    policy_scores = policy_out.squeeze(0)
    legal_moves = list(board_state.legal_moves)

    if not legal_moves:
        return None

    def move_priority(move: chess.Move) -> float:
        if not is_white_turn:
            mirrored_move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                move.promotion
            )
            idx = move_to_index(mirrored_move)
        else:
            idx = move_to_index(move)

        if idx is None:
            return float("-inf")
        return policy_scores[idx].item()

    legal_moves.sort(key=move_priority, reverse=True)
    legal_moves = legal_moves[:candidate_pruning]

    alpha, beta = float("-inf"), float("inf")
    best_val = float("-inf") if is_white_turn else float("inf")

    for move in legal_moves:
        board_state.push(move)
        val, _ = minimax(
            board_state,
            lookahead_depth - 1,
            alpha,
            beta,
            not is_white_turn,
            model,
            device
        )
        board_state.pop()

        if is_white_turn:
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
        else:
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, best_val)

    return best_move if best_move is not None else (legal_moves[0] if legal_moves else None)
