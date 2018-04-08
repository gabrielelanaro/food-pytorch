from mango import Experiment
from mango.trainer import MiniBatchTrainer

from .training.triplet import TripletModel
from .datasets.mock import MockFoodDataset

class Test(Experiment):

    model = TripletModel(cuda=False, embedding_size=64, margin=0.5, learning_rate=0.001)
    dataset = MockFoodDataset()
    trainer = MiniBatchTrainer(model, dataset, batch_size=32)
