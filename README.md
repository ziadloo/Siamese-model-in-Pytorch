# Siamese models in PyTorch

This repository holds a handful of classes that will facilitate implementing Siamese models and triplet loss functions in PyTorch. While the aim of these classes are Siamese models, but they could be used for any two-tower model as well.

To install:

```
pip install git+https://github.com/ziadloo/Siamese-model-in-Pytorch.git
```

To use:

```python
from Siamese import SiameseDataLoader, SiameseDataset, SiameseModel, triplet_loss


batch_size = 10
class_count = 10

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.912], std=[0.168]),
])

training_mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

train_samples = [[] for i in range(class_count)]
for X, y in training_mnist:
    train_samples[y].append(X)

training_generators = [
    torch.utils.data.TensorDataset(torch.stack(train_samples[i]))
    for i in range(class_count)
]
training_data = SiameseDataset(training_generators)
train_dataloader = SiameseDataLoader(
    training_data, batch_size=batch_size, shuffle=True
)
```

#### A complete example can be found in the "example" folder.
