# coding=utf-8
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class KNNSoftmax(nn.Module):
    def __init__(self, alpha=1, k=16, cuda=False):
        super(KNNSoftmax, self).__init__()
        self.alpha = alpha
        self.K = k
        self.cuda = cuda

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance
        dist_mat = euclidean_dist(inputs)
        if self.cuda:
            targets = targets.cuda()
        # split the positive and negative pairs

        eyes_ = Variable(torch.eye(n, n))
        if self.cuda:
            eyes_ = eyes_.cuda()

        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = (eyes_.eq(eyes_) - pos_mask)
        pos_mask = (pos_mask - eyes_.eq(1))

        loss = 0
        acc_num = 0

        total_positives = 0
        total_negatives = 0
        # Traverse each anchor, each sample act as an anchor to calculate the loss
        for i, dist in enumerate(dist_mat):

            # pos_pair is the distance of all positive samples of the i'th sample anchor
            pos_pair = dist[pos_mask[i]]

            # neg_pair is the distance of all negative samples whose ith sample is Anchor
            neg_pair = dist[neg_mask[i]]


            if len(pos_pair) == 0 or len(neg_pair) == 0:
                continue

            # Distance from the K+1 nearest neighbor to Anchor
            pair = torch.cat([pos_pair, neg_pair])

            threshold = torch.sort(pair)[0][self.K].data[0]

            # Get positive and negative pair of samples in K nearest neighbors
            
            pos_neig = pos_pair[(pos_pair < threshold).detach()]
            neg_neig = neg_pair[(neg_pair < threshold).detach()]

            # If there are no positive samples in the first K nearest neighbors, only the most recent positive sample is taken

            total_positives += len(pos_neig)
            
            if len(pos_neig) == 0:
                pos_neig = pos_pair[0]

            total_negatives += len(neg_neig)
            
            if len(neg_neig) > 0:
                neg_logit = torch.sum(torch.exp(-self.alpha*neg_neig))
            else:
                neg_logit = 0
                
            # The calculation of logit is to avoid floating point errors
            pos_logit = torch.sum(torch.exp(-self.alpha*pos_neig))
            loss_ = -torch.log(pos_logit/(pos_logit + neg_logit))

            if len(pos_neig) > len(neg_neig):
                acc_num += 1

            loss = loss + loss_

        loss = loss / n

        accuracy = float(acc_num)/n

        return loss, accuracy, total_positives/n, total_negatives/n


def euclidean_dist(inputs_):
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    # for numerical stability
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(KNNSoftmax(alpha=30)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
