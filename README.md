# Siamese models in PyTorch

This repository holds a handful of classes that will facilitate implementing Siamese models and triplet loss functions in PyTorch. While the aim of these classes are Siamese models, but they could be used for any two-tower model as well.

To install:

```
pip install git+https://github.com/ziadloo/Siamese-model-in-Pytorch.git
```

To use:

```python
from Siamese import SiameseDataLoader, SiameseSampleGenerator, SiameseDataset, SiameseModel, triplet_loss


class MnistGenerator(SiameseSampleGenerator):
    def __init__(self, ds, transform=None):
        self.samples = ds
        random.shuffle(self.samples)
        self.counter = 0
        self.transform = transform

    def shuffle(self):
        random.shuffle(self.samples)

    def __call__(self):
        curr = self.counter
        self.counter = (self.counter + 1) % len(self.samples)
        if self.transform is not None:
            return self.transform(self.samples[curr])
        else:
            return self.samples[curr]
```

A complete exmaple can be found in the "example" folder.
