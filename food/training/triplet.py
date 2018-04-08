import torch
import torch.nn as nn

from torch.autograd import Variable
import mango
from torchvision.models import resnet18, resnet34

from ..loss.triplet import OnlineTripletLoss
from ..loss.selectors import SemihardNegativeTripletSelector


RESNET_OUTPUT_SIZE = 1000

class Normalizer(nn.Module):
    
    def forward(self, emb):
        norms = torch.norm(emb, p=2, dim=1).data
        emb.data = emb.data.div(norms.view(-1,1).expand_as(emb))
        return emb


class TripletModel(mango.Model):

    cuda: bool
    embedding_size: int
    margin: float
    learning_rate: float
    checkpoint: str

    def initialize(self):
        self.net = nn.Sequential(
            resnet18(pretrained=False),
            nn.Linear(RESNET_OUTPUT_SIZE, self.embedding_size),
            Normalizer()
        )
        self.net.load_state_dict(torch.load(self.checkpoint))
        
        if self.cuda:
            self.net.cuda()

        self.criterion = OnlineTripletLoss(
            self.margin, 
            SemihardNegativeTripletSelector(self.margin))
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
        loss, n_triplets = self.criterion(embeddings, labels)
        
        if loss is not None:
            loss.backward()
            self.optimizer.step()
            self.net.eval()
            loss_value = loss.data[0]
        else:
            loss_value = 0.0
        
        self.reporter.add_scalar('loss', loss_value, step.global_step)
        self.reporter.add_scalar('n_triplets', n_triplets, step.global_step)
        
        if step.global_step % 100 == 0:
            self.save()

    def embed(self, images):
        images = Variable(torch.FloatTensor(images))
        self.net.eval()
        return self.net(images).data.cpu().numpy() 
        
    def epoch(self, step):
        self.save()
        self.reporter.log(f'epoch {step.epoch} completed')

    def save(self):
        self.reporter.log(f'Saving checkpoint {self.checkpoint}')
        torch.save(self.net.state_dict(), self.checkpoint)