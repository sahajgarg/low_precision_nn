import torch
import torch.nn as nn


class Einsum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, formula, x, y):
        return torch.einsum(formula, x, y)