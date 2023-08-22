import torch


def SiameseCollator(listOfTuples):
    s0 = torch.cat([torch.unsqueeze(t[0], dim=0) for t in listOfTuples])
    s1 = torch.cat([torch.unsqueeze(t[1], dim=0) for t in listOfTuples])
    return (s0, s1)

