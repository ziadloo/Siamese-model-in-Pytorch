import torch
from torch.utils.data import Dataset
from typing import List

# from .SiameseSampleGenerator import SiameseSampleGenerator


from torch import nn
import torchvision
import random
import copy
from torchvision import datasets
import torchvision.transforms as transforms


class SiameseDataset(Dataset):
    def __init__(self, generators: List[Dataset], *args, **kwargs):
        super(SiameseDataset, self).__init__(*args, **kwargs)
        self.size = min(len(g) for g in generators)
        self.generators = generators
        self.generator_indices = [
            torch.randperm(self.size).tolist() for _ in range(len(generators))
        ]

    def __len__(self):
        return self.size

    def shuffle(self):
        self.generator_indices = [
            torch.randperm(self.size).tolist() for _ in range(len(generators))
        ]

    def __getitem__(self, index):
        return (
            torch.stack(
                list(
                    g[self.generator_indices[i][index]][0]
                    for i, g in enumerate(self.generators)
                )
            ),
            torch.stack(
                list(
                    g[self.generator_indices[i][(index + 1) % self.size]][0]
                    for i, g in enumerate(self.generators)
                )
            ),
        )
