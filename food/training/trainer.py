import torch
from torch import FloatTensor
from torch.autograd import Variable

from ..models.siamese import make_siamese
from ..loss.contrastive import ContrastiveLoss

RESNET_INPUT_SIZE = 224

class Trainer:

    def __init__(self, cuda=False):
        self.cuda = cuda
        self.model = make_siamese()
        self.criterion = ContrastiveLoss()

        learning_rate = 0.01
        momentum = 0.9

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )


        if cuda:
            self.model.cuda()

        self.train_history = []

    def fit_batch(self, X_batch, y_batch):
        if not (X_batch.shape[3] == X_batch.shape[4] == RESNET_INPUT_SIZE):
            raise ValueError(f'Shape {X_batch.shape} is incompatible.\n'
                              'Dimensions 3 and 4 should be {RESNET_INPUT_SIZE}')

        X_0 = FloatTensor(X_batch[:, 0])
        X_1 = FloatTensor(X_batch[:, 1])
        labels = FloatTensor(y_batch)

        if self.cuda:
            X_0, X_1, labels = X_0.cuda(), X_1.cuda(), labels.cuda()

        X_0, X_1, labels = Variable(X_0), Variable(X_1), Variable(labels)
        output1, output2 = self.model(X_0, X_1)

        print("OUTPUT CALCULATED")
        loss = self.criterion(output1, output2, labels)
        self.train_history.append({'loss': loss.data[0]})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
