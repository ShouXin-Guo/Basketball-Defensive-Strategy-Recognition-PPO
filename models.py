"""Parallel 1D convolutional extractor and actor-critic policy."""
import torch
from torch import nn


def activation(name):
    return {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'prelu': lambda: nn.PReLU(), 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[name]()


def mlp(input_dim, depth, hidden, output_dim, act):
    blocks = []
    for _ in range(depth):
        blocks += [nn.Linear(input_dim, hidden), activation(act)]
        input_dim = hidden
    blocks.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*blocks)


class ConvBranch(nn.Module):
    def __init__(self, act):
        super().__init__()
        blocks = []
        c = 1
        for n, out in enumerate([32, 64, 128]):
            blocks.extend([nn.Conv1d(c, out, kernel_size=3, stride=1, padding=1), activation(act)])
            if n < 2:
                blocks.append(nn.MaxPool1d(2))
            c = out
        self.net = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.variant = cfg.model_variant
        if cfg.model_variant == 'no_cnn':
            self.encoder = nn.Identity()
            feature_size = 11
        else:
            self.branches = nn.ModuleList([ConvBranch(cfg.activation) for _ in range(cfg.branches)])
            feature_size = cfg.branches * 128
        self.policy = mlp(feature_size, cfg.policy_layers, cfg.policy_hidden, 2, cfg.activation)
        self.value = mlp(feature_size, cfg.value_layers, cfg.value_hidden, 1, cfg.activation)

    def forward(self, x):
        # Input is [batch,frames,11]; each branch scans the ordered feature axis.
        if self.variant == 'no_cnn':
            z = x.mean(dim=1)
        else:
            z = x.mean(dim=1).unsqueeze(1)
            z = torch.cat([branch(z) for branch in self.branches], dim=1)
        return self.policy(z), self.value(z).squeeze(-1)
