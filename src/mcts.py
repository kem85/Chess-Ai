"""
AlphaZero-style Monte Carlo Tree Search (MCTS) Engine for Chess.
Supports both PyTorch and ONNX Runtime backends.
"""

import math
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import torch
import chess

from .encoder import board_to_tensor, move_to_index

# Transposition & Evaluation Cache
evaluation_cache = {}
eval_cache_hits = 0
eval_cache_misses = 0


def mirror_move_for_black(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion
    )


def get_model_evaluation(
    model: Any,
    board: chess.Board,
    device: torch.device
) -> Tuple[torch.Tensor, float]:
    """
    Evaluates board position from the side-to-move perspective.
    Positive value means good for the player whose turn it is.
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
        value = float(v_out.item())
    else:
        input_tensor = torch.from_numpy(np_board).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode():
            policy_out, value_out = model(input_tensor)
        value = float(value_out.item())

    result = (policy_out, value)
    evaluation_cache[cache_key] = result

    if len(evaluation_cache) > 1_000_000:
        evaluation_cache.clear()

    return result


def terminal_value(board: chess.Board) -> Optional[float]:
    """Returns terminal value from side-to-move perspective."""
    if board.is_checkmate():
        return -1.0  # Current player is checkmated
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
        or board.is_repetition(3)
    ):
        return 0.0
    return None


def get_legal_move_priors(policy_logits: torch.Tensor, board: chess.Board) -> Dict[chess.Move, float]:
    """Calculates softmax prior probabilities over legal moves."""
    policy_scores = policy_logits.squeeze(0)
    legal_moves = list(board.legal_moves)

    move_priors = {}
    indexed = []

    for move in legal_moves:
        if board.turn == chess.BLACK:
            idx = move_to_index(mirror_move_for_black(move))
        else:
            idx = move_to_index(move)

        if idx is None:
            continue

        score = policy_scores[idx].item()
        indexed.append((move, score))

    if not indexed:
        uniform = 1.0 / max(1, len(legal_moves))
        for move in legal_moves:
            move_priors[move] = uniform
        return move_priors

    logits = torch.tensor([score for _, score in indexed], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0).tolist()

    for (move, _), prob in zip(indexed, probs):
        move_priors[move] = float(prob)

    return move_priors


class MCTSNode:
    """MCTS Node holding prior probability, visit count, and cumulative value sum."""
    __slots__ = (
        "board",
        "parent",
        "move",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "is_expanded",
    )

    def __init__(self, board: chess.Board, parent=None, move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = float(prior)
        self.children: Dict[chess.Move, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, model: Any, device: torch.device) -> float:
        term = terminal_value(self.board)
        if term is not None:
            self.is_expanded = True
            return term

        policy_logits, value = get_model_evaluation(model, self.board, device)
        priors = get_legal_move_priors(policy_logits, self.board)

        for move, prior in priors.items():
            next_board = self.board.copy(stack=False)
            next_board.push(move)
            self.children[move] = MCTSNode(
                board=next_board,
                parent=self,
                move=move,
                prior=prior,
            )

        self.is_expanded = True
        return value

    def select_child(self, c_puct: float = 1.5) -> "MCTSNode":
        best_score = float("-inf")
        best_child = None

        sqrt_parent_visits = math.sqrt(max(1, self.visit_count))

        for child in self.children.values():
            q = -child.q_value  # Perspective flip for child node
            u = c_puct * child.prior * (sqrt_parent_visits / (1 + child.visit_count))
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


def backpropagate(search_path: List[MCTSNode], value: float):
    """Backpropagates leaf evaluation up the tree, alternating signs."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def add_dirichlet_noise(root: MCTSNode, alpha: float = 0.3, epsilon: float = 0.25):
    """Adds Dirichlet exploration noise to root legal move priors."""
    if not root.children:
        return

    moves = list(root.children.keys())
    noise = np.random.dirichlet([alpha] * len(moves))

    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = (1 - epsilon) * child.prior + epsilon * float(n)


def run_mcts(
    board: chess.Board,
    model: Any,
    device: torch.device,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> MCTSNode:
    root = MCTSNode(board.copy(stack=False))
    root_value = root.expand(model, device)
    root.visit_count = 1
    root.value_sum = root_value

    if add_noise:
        add_dirichlet_noise(root, alpha=dirichlet_alpha, epsilon=dirichlet_epsilon)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.is_expanded and node.children:
            node = node.select_child(c_puct=c_puct)
            search_path.append(node)

        if terminal_value(node.board) is not None:
            value = terminal_value(node.board)
        else:
            value = node.expand(model, device)

        backpropagate(search_path, value)

    return root


def get_best_move_mcts(
    board_state: chess.Board,
    model: Any,
    device: torch.device,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    temperature: float = 0.0,
    add_noise: bool = False,
) -> Optional[chess.Move]:
    """Finds best move via Monte Carlo Tree Search simulations."""
    root = run_mcts(
        board=board_state,
        model=model,
        device=device,
        num_simulations=num_simulations,
        c_puct=c_puct,
        add_noise=add_noise,
    )

    if not root.children:
        return None

    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

    if temperature <= 1e-8:
        return moves[int(np.argmax(visits))]

    visits = visits ** (1.0 / temperature)
    probs = visits / visits.sum()
    choice = np.random.choice(len(moves), p=probs)
    return moves[int(choice)]
