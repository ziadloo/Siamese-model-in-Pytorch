from torch.utils.data import DataLoader
from .SiameseDataset import SiameseDataset
from .SiameseBatchSampler import SiameseBatchSampler


class SiameseDataLoader(DataLoader):
    def __init__(
        self,
        dataset: SiameseDataset,
        batch_size: int,
        shuffle: bool = False,
        *args,
        **kwargs
    ):
        super(SiameseDataLoader, self).__init__(
            dataset,
            batch_sampler=SiameseBatchSampler(len(dataset), batch_size),
            *args,
            **kwargs
        )
