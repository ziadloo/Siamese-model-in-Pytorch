import torch
from torch.utils.data import Dataset
from typing import List
from torch.utils.data import DataLoader
from .SiameseSampleGenerator import SiameseSampleGenerator


class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, size: int, generators: List[SiameseSampleGenerator], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.generators = generators

    def __len__(self):
        return self.size
    
    def shuffle(self):
        for g in self.generators:
            g.shuffle()

    def __getitem__(self, index):
        return (self.generators[index](), self.generators[index]())
