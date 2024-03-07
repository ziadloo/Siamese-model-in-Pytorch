import torch


def info_nce_loss(v1, v2):
    # similarity scores
    scores = torch.mm(v1, v2.t())
    # Labels is a vector holding the diagonal indices
    labels = torch.arange(scores.size(0)).to(scores.device)

    return torch.nn.functional.cross_entropy(scores, labels)
