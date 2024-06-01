import torch


def triplet_loss(v1, v2, margin=1.0):
    scores = torch.matmul(v1, v2.permute(*range(v2.dim() - 2), -1, -2))
    class_size = scores.size(-2)
    positive = scores.diagonal(dim1=-2, dim2=-1)
    negative_without_positive = scores - (
        torch.eye(class_size, device=scores.device) * 2.0
    )
    closest_negative, _ = torch.max(negative_without_positive, dim=-1)
    negative_zero_on_duplicate = scores * (
        1.0 - torch.eye(class_size, device=scores.device)
    )
    mean_negative = torch.sum(negative_zero_on_duplicate, dim=-1) / (class_size - 1)
    triplet_loss1 = torch.clamp(margin + closest_negative - positive, min=0.0)
    triplet_loss2 = torch.clamp(margin + mean_negative - positive, min=0.0)
    triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)
    return triplet_loss.mean()
