import torch


def info_nce_loss(v1, v2):
    # similarity scores
    scores = torch.matmul(v1, v2.permute(*range(v2.dim() - 2), -1, -2))
    # Labels is a vector holding the diagonal indices
    labels = torch.arange(scores.size(-1)).to(scores.device)
    if scores.dim() - labels.dim() > 1:
        labels = labels.expand([*scores.shape[:-2], -1])
    return torch.nn.functional.cross_entropy(scores, labels)
