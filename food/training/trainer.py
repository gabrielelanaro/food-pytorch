import torch
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..models.siamese import make_siamese
from ..loss.contrastive import ContrastiveLoss

RESNET_INPUT_SIZE = 224

class Trainer:

    def __init__(self, cuda=False, lr=0.001):
        self.cuda = cuda
        self.model = make_siamese()
        self.criterion = ContrastiveLoss()

        learning_rate = lr
        momentum = 0.9

        self.optimizer = torch.optim.Adam(
            list(self.model.distance_network.parameters())+
            list(self.model.encoder.parameters()),
            lr=learning_rate
        )

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        if cuda:
            self.model.cuda()

        self.train_history = []

    def adapt_image(self, X):
        pass

    def fit_batch(self, X_batch, y_batch):
        if not (X_batch.shape[3] == X_batch.shape[4] == RESNET_INPUT_SIZE):
            raise ValueError(f'Shape {X_batch.shape} is incompatible.\n'
                             f'Dimensions 3 and 4 should be {RESNET_INPUT_SIZE}')
    
        self.optimizer.zero_grad()
        self.model.train()
        X_0 = FloatTensor(X_batch[:, 0])
        X_1 = FloatTensor(X_batch[:, 1])
        labels = FloatTensor(y_batch)

        if self.cuda:
            X_0, X_1, labels = X_0.cuda(), X_1.cuda(), labels.cuda()

        X_0, X_1, labels = Variable(X_0), Variable(X_1), Variable(labels)
        output1, output2 = self.model(X_0, X_1)

        loss = self.criterion(output1, output2, labels)
        self.train_history.append({'loss': loss.data[0]})
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    def predict(self, X):
        self.model.eval()
        X_batch = X
        X_0 = FloatTensor(X_batch[:, 0])
        X_1 = FloatTensor(X_batch[:, 1])
        
        if self.cuda:
            X_0, X_1 = X_0.cuda(), X_1.cuda()

        X_0, X_1 = Variable(X_0), Variable(X_1)

        output1, output2 = self.model(X_0, X_1)
        return torch.sum(torch.pow(output1 - output2, 2), 1).cpu().data.numpy()

def _tensors_equal(a, b):
    return (a.cpu().data.numpy() == b.cpu().data.numpy()).all()