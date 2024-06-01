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
        self.shifted_generator_indices = [
            gi[1:] + gi[:1] for gi in self.generator_indices
        ]

    def __len__(self):
        return self.size

    def shuffle(self):
        self.generator_indices = [
            torch.randperm(self.size).tolist() for _ in range(len(self.generators))
        ]
        self.shifted_generator_indices = [
            gi[1:] + gi[:1] for gi in self.generator_indices
        ]

    def __getitem__(self, index):
        return (
            torch.cat(
                list(
                    g[self.generator_indices[i][index]][0].unsqueeze(dim=-4)
                    for i, g in enumerate(self.generators)
                ),
                dim=-4,
            ),
            torch.cat(
                list(
                    g[self.shifted_generator_indices[i][index]][0].unsqueeze(dim=-4)
                    for i, g in enumerate(self.generators)
                ),
                dim=-4,
            ),
        )
