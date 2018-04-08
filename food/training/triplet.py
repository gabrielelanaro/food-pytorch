import torch
import torch.nn as nn

from torch.autograd import Variable
import mango
from torchvision.models import resnet18, resnet34

from ..loss.triplet import OnlineTripletLoss
from ..loss.selectors import SemihardNegativeTripletSelector


RESNET_OUTPUT_SIZE = 1000

class TripletModel(mango.Model):

    cuda: bool
    embedding_size: int
    margin: float
    learning_rate: float

    def initialize(self):
        self.net = nn.Sequential(
            resnet18(pretrained=False),
            nn.Linear(RESNET_OUTPUT_SIZE, self.embedding_size)
        )

        self.criterion = OnlineTripletLoss(self.margin, SemihardNegativeTripletSelector(self.margin))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def batch(self, batch, step):
        images = Variable(torch.FloatTensor(batch['images']))
        labels = Variable(torch.FloatTensor(batch['labels'].astype('float')))
        self.reporter.add_scalar('step', step.step, step.global_step)
        self.optimizer.zero_grad()
        self.net.train()

        if self.cuda:
            images.cuda()

        embeddings = self.net(images)
        loss, n_triplets = self.criterion(embeddings, labels)

        loss.backward()
        self.optimizer.step()
        self.net.eval()

        self.reporter.add_scalar('loss', loss.data[0], step.global_step)
        self.reporter.add_scalar('n_triplets', n_triplets, step.global_step)

    def epoch(self, step):
        torch.save(self.net.state_dict(), 'fine_tuned2.torch')
