from toolz import pluck
import numpy as np
import mango

class MockFoodDataset(mango.SplitDataset):

    def build(self):
        self.train_images = np.random.rand(1000, 3, 224, 224)
        self.train_labels = np.random.randint(0, 3, size=1000)
        
        self.test_images = np.random.rand(512, 3, 224, 224)
        self.test_labels = np.random.randint(0, 3, size=512)

        self.train_data = list(zip(self.train_images, self.train_labels))
        self.test_data = list(zip(self.test_images, self.test_labels))

    def train(self):
        return self.train_data
    
    def test(self):
        return self.test_data
    
    def transform_train(self, data):
        return {'images': np.array(list(pluck(0, data))),
                'labels': np.array(list(pluck(1, data)))}
    
    transform_test = transform_train