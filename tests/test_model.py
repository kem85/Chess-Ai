import pytest
import torch
from src.model import ChessResNet, ResBlock


def test_resblock_output_shape():
    block = ResBlock(channels=64)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)


def test_chess_resnet_forward_pass():
    model = ChessResNet(num_blocks=2, hidden_channels=32)
    model.eval()
    x = torch.randn(4, 12, 8, 8)

    with torch.no_grad():
        policy, value = model(x)

    assert policy.shape == (4, 4864)
    assert value.shape == (4, 1)
    assert torch.all((value >= -1.0) & (value <= 1.0))
