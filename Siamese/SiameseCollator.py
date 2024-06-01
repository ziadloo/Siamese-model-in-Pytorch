import torch


def SiameseCollator(listOfTuples):
    s0 = torch.stack([t[0] for t in listOfTuples])
    s1 = torch.stack([t[1] for t in listOfTuples])
    return (s0, s1)
