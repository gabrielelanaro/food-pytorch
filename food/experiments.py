from mango import Experiment
from mango.trainer import MiniBatchTrainer
from mango import reporters
from mango.loaders import MiniBatchLoader

from .models.triplet import TripletModel
from .datasets.mock import MockFoodDataset
from .datasets.food import Food101Dataset

reporter = reporters.CombinedReporter([
    reporters.TextReporter(),
    reporters.LogReporter('logs/test')
])


class Test(Experiment):

    model = TripletModel(cuda=False,
                         embedding_size=64,
                         margin=0.5,
                         learning_rate=0.001,
                         reporter=reporter,
                         checkpoint='test.params')
    dataset = MockFoodDataset()
    loader = MiniBatchLoader(dataset, batch_size=8)
    trainer = MiniBatchTrainer(model, loader)


reporter = reporters.CombinedReporter([
    reporters.TextReporter(),
    reporters.LogReporter('logs/main')
])

class Main(Experiment):

    model = TripletModel(cuda=True,
                         embedding_size=64,
                         margin=0.5,
                         learning_rate=0.001,
                         checkpoint='main.params',
                         reporter=reporter)
    dataset = Food101Dataset('/home/paperspace/data/food-101/food-101/')
    loader = MiniBatchLoader(dataset, batch_size=32)
    trainer = MiniBatchTrainer(model, loader)
