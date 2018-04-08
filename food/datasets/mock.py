import numpy as np

class MockFoodDataset:

    def load(self):
        pass

    def __getitem__(self, index):
        return {'images': np.random.rand(3, 224, 224), 'labels': np.random.randint(10)}

    def __len__(self):
        return 1000
