import torch
from .SiameseDataset import SiameseDataset


class SiameseBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
    ):
        super(SiameseBatchSampler, self).__init__([], batch_size, drop_last=False)
        indices = torch.arange(dataset_size).tolist()
        self.batched_indices = [
            indices[idx : idx + batch_size]
            for idx in range(0, len(indices), batch_size)
        ]

    def __iter__(self):
        return (i for i in self.batched_indices)

    def __len__(self):
        return len(self.batched_indices)
