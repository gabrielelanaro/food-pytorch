import os
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
import mango
from torchvision.models import resnet18, resnet34
from toolz import partition_all

from ..loss.triplet import OnlineTripletLoss
from ..loss.KNNSoftmax import KNNSoftmax
from ..loss.selectors import SemihardNegativeTripletSelector

RESNET_OUTPUT_SIZE = 1000

class Normalizer(nn.Module):

    def forward(self, emb):
        norms = torch.norm(emb, p=2, dim=1)
        emb = emb.div(norms.view(-1,1).expand_as(emb))

        return emb


class TripletModel(mango.Model):

    cuda = mango.Param(bool)
    embedding_size = mango.Param(int)
    margin = mango.Param(float)
    learning_rate = mango.Param(float)
    checkpoint = mango.Param(str)

    def build(self):
        self.net = nn.Sequential(
            resnet18(pretrained=False),
            nn.Linear(RESNET_OUTPUT_SIZE, self.embedding_size),
            Normalizer()
        )
        if os.path.exists(self.checkpoint):
            self.reporter.log(f'Loading checkpoint {self.checkpoint}')
            self.net.load_state_dict(torch.load(self.checkpoint))
        
        if self.cuda:
            self.net.cuda()

        # self.criterion = OnlineTripletLoss(
        #     self.margin,
        #     SemihardNegativeTripletSelector(self.margin))

        self.criterion = KNNSoftmax(k=4)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate
        )
                

    def batch(self, batch, step):
        self.reporter.log(f'Step {step.step} of {step.max_steps}')
        images = Variable(torch.FloatTensor(batch['images']))
        labels = Variable(torch.FloatTensor(batch['labels'].astype('float')))

        if len(images.size()) == 1:
            self.reporter.log(f'What the hell {images}\n')
            return

        self.optimizer.zero_grad()
        self.net.train()

        if self.cuda:
            images = images.cuda()
            labels = labels.cuda()

        embeddings = self.net(images)
        # loss, n_triplets = self.criterion(embeddings, labels)
        loss, accuracy, n_pos, n_neg = self.criterion(embeddings, labels)
        if loss is not None:
            loss.backward()
            self.optimizer.step()
            self.net.eval()
            loss_value = loss.data.cpu().numpy()[0]
        else:
            loss_value = 0.0

        self.reporter.add_scalar('loss', loss_value, step.global_step)
        self.reporter.add_scalar('accuracy', accuracy, step.global_step)
        self.reporter.add_scalar('n_pos', n_pos, step.global_step)
        self.reporter.add_scalar('n_neg', n_neg, step.global_step)

        if step.global_step % 100 == 0:
            self.save()

    def embed(self, images):
        images = Variable(torch.FloatTensor(images))
        self.net.eval()
        return self.net(images).data.cpu().numpy()

    def epoch(self, step, dataset):
        self.save()
        
        if step.epoch == 0:
            self.validation_ix = np.random.randint(0, len(dataset), size=100) 
        
        data = [dataset.get(ix, 'test') for ix in self.validation_ix]
        
        labels = []
        embs = []
        for batch in partition_all(64, data):
            images = np.array([d['images'] for d in batch])
            images = Variable(torch.FloatTensor(images))
            if self.cuda:
                images = images.cuda()
            embs.extend(self.net(images).data.cpu().numpy())
            labels.extend(d['labels'] for d in batch)
        
        embs = np.array(embs)
        self.reporter.add_embedding('embeddings', embs, labels=labels, iteration=step.epoch)
        self.reporter.add_histogram('embeddings', embs, iteration=step.epoch)
        
        self.reporter.log(f'epoch {step.epoch} completed')

    def save(self):
        self.reporter.log(f'Saving checkpoint {self.checkpoint}')
        torch.save(self.net.state_dict(), self.checkpoint)
