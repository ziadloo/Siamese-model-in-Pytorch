import torch


def triplet_loss(v1, v2, margin=1.0):
    scores = torch.mm(v1, v2.t())
    batch_size = scores.size(0)
    positive = scores.diag()
    negative_without_positive = scores - (torch.eye(batch_size, device=scores.device) * 2.0)
    closest_negative, _ = torch.max(negative_without_positive, dim=1)
    negative_zero_on_duplicate = scores * (1.0 - torch.eye(batch_size, device=scores.device))
    mean_negative = torch.sum(negative_zero_on_duplicate, dim=1) / (batch_size - 1)
    triplet_loss1 = torch.clamp(margin + closest_negative - positive, min=0.0)
    triplet_loss2 = torch.clamp(margin + mean_negative - positive, min=0.0)
    triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)
    return triplet_loss
