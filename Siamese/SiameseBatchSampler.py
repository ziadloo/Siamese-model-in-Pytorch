import torch
from .SiameseDataset import SiameseDataset


class SiameseBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, ds: SiameseDataset, batch_size: int, class_count: int, iteration_count: int):
        super().__init__([], batch_size, drop_last=False)
        # Check if n is greater than m, which would make it impossible to generate unique numbers
        if batch_size > class_count:
            raise ValueError("batch_size should be less than or equal to class_count for generating unique numbers.")
        self.ds = ds
        self.batch_size = batch_size
        self.class_count = class_count
        self.iteration_count = iteration_count

    def __iter__(self):
        for i in range(self.iteration_count):
            # Generate a sequence of integers from 0 to self.class_count (exclusive)
            sequence = torch.arange(self.class_count)

            # Shuffle the sequence to get a random order
            shuffled_sequence = torch.randperm(self.class_count)

            # Select the first self.batch_size elements from the shuffled sequence
            unique_random_list = shuffled_sequence[:self.batch_size]

            yield unique_random_list.tolist()
        self.ds.shuffle()
