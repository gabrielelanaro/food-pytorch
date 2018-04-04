import torch
import torch.nn


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        positive_increase = 10.0
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        
        mdist_clamped = torch.clamp(self.margin - dist, min=0)
        matching_term = torch.pow(dist, 2)
        nonmatching_term = torch.pow(mdist_clamped,2)

        loss = (1-y) * nonmatching_term + y * matching_term 
        loss = torch.mean(loss) / 2.0
        return loss

def isnan(x):
     return torch.sum(x != x).cpu().data.numpy()[0]