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
from ..cls import CyclicLR

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
    k = mango.Param(int, default=4)
    dropout = mango.Param(float, default=0.2)

    def build(self):
        self.net = nn.Sequential(
            resnet18(pretrained=False),
            nn.Dropout(self.dropout),
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

        self.criterion = KNNSoftmax(self.k, cuda=self.cuda)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate
        )
        
        if os.path.exists(self.checkpoint + '.adam'):
            self.reporter.log(f'Loading adam checkpoint {self.checkpoint}.adam')
            self.optimizer.load_state_dict(torch.load(self.checkpoint + '.adam'))

        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # self.scheduler = CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.001, step_size=200)
        
        
        

    def batch(self, batch, step):
        self.reporter.log(f'Step {step.step} of {step.max_steps}')
        #self.scheduler.batch_step()
        images = Variable(torch.FloatTensor(batch['images']))
        labels = Variable(torch.FloatTensor(batch['labels'].astype('float')))
        
        if len(images.size()) == 1:
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

    def embed(self, images):
        images = Variable(torch.FloatTensor(images))
        self.net.eval()
        return self.net(images).data.cpu().numpy()

    def epoch(self, step, loader):
        self.save()
        
        self.net.eval()
        loader.test()
        
        label_names = []
        embs = []
        loss = []
        accuracies = []
        
        for batch in loader:
            
            images = Variable(torch.FloatTensor(batch['images']))
            lbl = np.array([hash(l) for l in batch['labels']])
            lbl = Variable(torch.FloatTensor(lbl.astype('float')))

            if self.cuda:
                images = images.cuda()
            
            embeddings = self.net(images)
            loss_, acc, *others = self.criterion(embeddings, lbl)
            loss.append(loss_.data.cpu().numpy()[0])
            
            embs.extend(embeddings.data.cpu().numpy())
            label_names.extend(batch['labels'])
            accuracies.append(acc)

        embs = np.array(embs)
        
        self.reporter.add_scalar('val_loss', np.mean(loss), step.global_step)
        self.reporter.add_scalar('val_acc', np.mean(acc), step.global_step)
        self.reporter.add_embedding('embeddings', embs, labels=label_names, iteration=step.epoch)
        self.reporter.add_histogram('embeddings', embs, iteration=step.epoch)
        
        self.reporter.log(f'epoch {step.epoch} completed')

    def save(self):
        self.reporter.log(f'Saving checkpoint {self.checkpoint}')
        torch.save(self.net.state_dict(), self.checkpoint)
        torch.save(self.optimizer.state_dict(), self.checkpoint + '.adam')
        
        
