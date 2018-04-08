from mango import Experiment
from mango.trainer import MiniBatchTrainer
from mango import reporters


from .training.triplet import TripletModel
from .datasets.mock import MockFoodDataset
from .datasets.food import Food101Dataset

reporter = reporters.CombinedReporter([
    reporters.TextReporter(),
    reporters.LogReporter('logs/test')
])


class Test(Experiment):

    model = TripletModel(cuda=True, 
                         embedding_size=64, 
                         margin=0.5, 
                         learning_rate=0.001,
                         reporter=reporter,
                         checkpoint='test.params')
    dataset = MockFoodDataset()
    trainer = MiniBatchTrainer(model, dataset, batch_size=32)


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
    trainer = MiniBatchTrainer(model, dataset, batch_size=128)