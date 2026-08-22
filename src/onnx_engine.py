"""
High-Performance ONNX Runtime Engine for Chess Neural Network Inference.
"""

from typing import Tuple, Optional
import os
import numpy as np
import chess

from .encoder import board_to_tensor

# Transposition cache for ONNX evaluations
onnx_eval_cache = {}


class ONNXChessModel:
    """
    Wrapper for ONNX Runtime Inference Session supporting FP32, FP16, and INT8 models.
    Automatically prioritizes CUDAExecutionProvider if GPU is available, otherwise CPU.
    """

    def __init__(self, onnx_path: str, use_gpu: bool = True):
        import onnxruntime as ort

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

        self.onnx_path = onnx_path
        
        # Configure session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.log_severity_level = 3  # Warning level only

        providers = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }
            ))
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.policy_name = self.session.get_outputs()[0].name
        self.value_name = self.session.get_outputs()[1].name
        
        # Check input precision (FP32 vs FP16)
        self.input_type = self.session.get_inputs()[0].type

    def __call__(self, input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on (1, 12, 8, 8) input numpy array.
        Returns:
            - policy_logits: numpy array of shape (1, 4864)
            - value: numpy array of shape (1, 1)
        """
        if "float16" in self.input_type:
            input_array = input_array.astype(np.float16)
        else:
            input_array = input_array.astype(np.float32)

        outputs = self.session.run(
            [self.policy_name, self.value_name],
            {self.input_name: input_array}
        )
        return outputs[0], outputs[1]


def get_onnx_evaluation(
    onnx_model: ONNXChessModel,
    board: chess.Board
) -> Tuple[np.ndarray, float]:
    """
    Evaluates board position with ONNX Runtime.
    Returns (policy_logits, value_score_from_side_to_move_perspective).
    """
    board_key = board._transposition_key()
    cache_key = (id(onnx_model), board_key)

    if cache_key in onnx_eval_cache:
        return onnx_eval_cache[cache_key]

    is_black = (board.turn == chess.BLACK)
    eval_board = board.mirror() if is_black else board

    np_board = board_to_tensor(eval_board)
    input_tensor = np.expand_dims(np_board, axis=0)

    policy_out, value_out = onnx_model(input_tensor)
    value = float(value_out.item())

    result = (policy_out, value)
    onnx_eval_cache[cache_key] = result

    if len(onnx_eval_cache) > 1_000_000:
        onnx_eval_cache.clear()

    return result
