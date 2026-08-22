import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Block with two 3x3 Conv2d layers and BatchNorm."""

    def __init__(self, channels: int = 128):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessResNet(nn.Module):
    """
    Dual-Head Chess Deep Residual Neural Network.
    
    Inputs:
        - Board tensor representation: (B, 12, 8, 8)
    Outputs:
        - Policy logits: (B, 4864) corresponding to 76 move planes x 64 squares
        - Value evaluation: (B, 1) in range [-1.0, 1.0] (tanh)
    """

    def __init__(self, num_blocks: int = 10, hidden_channels: int = 128):
        super(ChessResNet, self).__init__()

        self.start_conv = nn.Conv2d(12, hidden_channels, kernel_size=3, padding=1)
        self.start_bn = nn.BatchNorm2d(hidden_channels)
        self.res_blocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(num_blocks)])

        # --- Policy Head (76 Move Planes x 8 x 8 squares = 4864 classes) ---
        self.policy_conv = nn.Conv2d(hidden_channels, 76, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(76)
        self.policy_fc = nn.Linear(76 * 8 * 8, 4864)

        # --- Value Head (Position Evaluation) ---
        self.eval_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.eval_bn = nn.BatchNorm2d(1)
        self.eval_fc1 = nn.Linear(8 * 8, 64)
        self.eval_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.start_bn(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy output
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 4864)
        policy_out = self.policy_fc(p)

        # Value output
        v = F.relu(self.eval_bn(self.eval_conv(x)))
        v = v.view(-1, 64)
        v = F.relu(self.eval_fc1(v))
        v = self.eval_fc2(v)
        value_out = torch.tanh(v)

        return policy_out, value_out
