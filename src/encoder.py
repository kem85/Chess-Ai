from typing import Optional
import numpy as np
import torch
import chess


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encodes a python-chess Board into a 12x8x8 float32 numpy tensor.
    Channels 0-5: White (P, N, B, R, Q, K)
    Channels 6-11: Black (P, N, B, R, Q, K)
    """
    tensor = np.zeros((12, 64), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        layer = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[layer][sq] = 1.0

    tensor = tensor.reshape(12, 8, 8)
    return np.flip(tensor, axis=1).copy()


def mirror_board_for_black(board_tensor: torch.Tensor) -> torch.Tensor:
    """
    Mirrors spatial rows and swaps White/Black color channels for black-to-move perspective.
    """
    flipped_spatial = torch.flip(board_tensor, dims=[2])
    white_channels = flipped_spatial[:, 0:6, :, :]
    black_channels = flipped_spatial[:, 6:12, :, :]
    return torch.cat((black_channels, white_channels), dim=1)


def move_to_index(move: chess.Move) -> Optional[int]:
    """
    Maps a chess.Move to an index in [0, 4863] based on 76 move planes:
      - 56 planes: Queen-like ray moves (8 directions x 7 distances)
      - 8 planes: Knight moves
      - 12 planes: Underpromotions (3 directions x 4 promotion pieces)
    """
    from_sq = move.from_square
    to_sq = move.to_square

    r0, c0 = divmod(from_sq, 8)
    r1, c1 = divmod(to_sq, 8)

    dr = r1 - r0
    dc = c1 - c0

    # 1. Knight moves (8 planes: 56 to 63)
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    if (dr, dc) in knight_moves:
        plane = 56 + knight_moves.index((dr, dc))
        return plane * 64 + from_sq

    # 2. Pawn promotions (12 planes: 64 to 75)
    if move.promotion is not None:
        if dc == 0:
            dir_idx = 0
        elif dc == -1:
            dir_idx = 1
        else:
            dir_idx = 2

        promo_map = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 3,
        }
        piece_idx = promo_map[move.promotion]
        plane = 64 + dir_idx * 4 + piece_idx
        return plane * 64 + from_sq

    # 3. Queen-like sliding moves (56 planes: 0 to 55)
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
    for dir_idx, (drd, dcd) in enumerate(directions):
        for dist in range(1, 8):
            if dr == drd * dist and dc == dcd * dist:
                plane = dir_idx * 7 + (dist - 1)
                return plane * 64 + from_sq

    return None
