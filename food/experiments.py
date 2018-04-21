from mango import Experiment
from mango.trainer import MiniBatchTrainer
from mango import reporters
<<<<<<< Updated upstream
from mango.loaders import MiniBatchLoader

from .models.triplet import TripletModel
=======
from mango.reporters.tensorboard import TensorboardReporter

from .training.triplet import TripletModel
>>>>>>> Stashed changes
from .datasets.mock import MockFoodDataset
from .datasets.food import Food101Dataset

reporter = reporters.CombinedReporter([
    reporters.TextReporter(),
    reporters.LogReporter('logs/test'),
    TensorboardReporter('runs/test')
])


class Test(Experiment):

    model = TripletModel(cuda=True, 
                         embedding_size=32, 
                         margin=0.5, 
                         learning_rate=0.05,
                         reporter=reporter,
                         checkpoint='test.params')
    dataset = MockFoodDataset()
    loader = MiniBatchLoader(dataset, batch_size=8)
    trainer = MiniBatchTrainer(model, loader)


reporter = reporters.CombinedReporter([
    reporters.TextReporter(),
    reporters.LogReporter('logs/main'),
    TensorboardReporter('runs/main')
])

class Main(Experiment):

    model = TripletModel(cuda=True,
                         embedding_size=4,
                         margin=0.25,
                         learning_rate=0.0001,
                         checkpoint='main.params',
                         reporter=reporter)
    dataset = Food101Dataset('/home/paperspace/data/food-101/food-101/', classes=['steak', 'oysters', 'sashimi', 'omelette', 'pizza', 'bruschetta'])
    loader = MiniBatchLoader(dataset, batch_size=32)
    trainer = MiniBatchTrainer(model, loader)
