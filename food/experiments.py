from mango import Experiment
from mango.trainer import MiniBatchTrainer
from mango import reporters
from mango.loaders import MiniBatchLoader

from .models.triplet import TripletModel
from mango.reporters.tensorboard import TensorboardReporter

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
                         k=16,
                         checkpoint='test.params')
    dataset = MockFoodDataset()
    loader = MiniBatchLoader(dataset, batch_size=128)
    trainer = MiniBatchTrainer(model, loader)


reporter = reporters.CombinedReporter([
    reporters.TextReporter(),
    reporters.LogReporter('logs/main'),
    TensorboardReporter('runs/main')
])

class Main(Experiment):

    model = TripletModel(cuda=True,
                         embedding_size=16,
                         margin=0.25,
                         learning_rate=0.001,
                         checkpoint='main_knn.params',
                         reporter=reporter,
                         k=32)
    dataset = Food101Dataset('/home/paperspace/data/food-101/food-101/', classes=['steak', 
                                                                                  'oysters', 
                                                                                  'sashimi', 
                                                                                  'omelette', 
                                                                                  'pizza', 
                                                                                  'bruschetta', 
                                                                                  'foie_gras', 
                                                                                  'pork_chop',
                                                                                  'tiramisu',
                                                                                  'baklava',
                                                                                  'apple_pie',
                                                                                  'escargot',
                                                                                  'baklava',
                                                                                  'gnocchi',
                                                                                  'spaghetti_bolognese',
                                                                                  'scallops',
                                                                                  'mussels'])
    loader = MiniBatchLoader(dataset, batch_size=128)
    trainer = MiniBatchTrainer(model, loader, epochs=1000)
