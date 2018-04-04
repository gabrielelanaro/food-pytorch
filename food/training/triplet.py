import torch

from tensorflowX import SummaryWriter
from .base import Trainable
from ..loss.triplet import OnlineTripletLoss
from ..loss.selectors import SemihardNegativeTripletSelector


RESNET_OUTPUT_SIZE = 1000

class TripletTrainer(Trainable):

    def __init__(self,
                 dataset,
                 cuda=False,
                 embedding_size=64,
                 margin=1.0,
                 learning_rate=0.001):
        super().__init__(dataset)
        self.model = Sequential(
            resnet18(pretrained=True),
            nn.Linear(RESNET_OUTPUT_SIZE, embedding_size)
        )
        self.criterion = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()
        self.cuda = cuda

    def do_batch(self, batch, global_step):
        images = batch['images']
        labels = batch['labels']

        self.optimizer.zero_grad()
        self.model.train()

        if self.cuda:
            images.cuda()

        embeddings = self.model(images)
        loss, n_triplets = self.criterion(embeddings, labels)
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        if global_step % 100 == 0:
            self.writer.add_scalar('loss', loss.data[0], global_step)
            self.writer.add_scalar('n_triplets', n_triplets, global_step)

    def do_epoch(self, global_step):
        torch.save(self.model.state_dict(), 'fine_tuned2.torch')
