from torch.utils.data import DataLoader
from .SiameseDataset import SiameseDataset
from .SiameseCollator import SiameseCollator
from .SiameseBatchSampler import SiameseBatchSampler


class SiameseDataLoader(DataLoader):
    def __init__(self, dataset: SiameseDataset, batch_size: int, class_count: int, *args, **kwargs):
        super().__init__(dataset,
                         collate_fn=SiameseCollator,
                         batch_sampler=SiameseBatchSampler(dataset, batch_size, class_count, len(dataset)),
                         *args, **kwargs)

