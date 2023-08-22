import torch
from torch import nn


class SiameseModel(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, inputs):
        v1 = self.tower(inputs[0])
        v2 = self.tower(inputs[1])
        return (v1, v2)
