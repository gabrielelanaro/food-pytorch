import os
import glob
import numpy as np
import mango
from toolz import concat
from skimage import io, transform
from torch.utils.data import Dataset

from toolz.curried import curry, pluck
from ..transformers.pairwise import PairwiseTransform
import random
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import multiprocessing

def load_filenames(path, classes=None):
    '''
    Create a dataset where to a picture we have an associated label
    '''
    
    if classes is None:
        classes = os.listdir(os.path.join(path, 'images'))
    else:
        classes = classes
    
    dataset = concat([[(c, img) for img in glob.glob(os.path.join(path, 'images', c, '*.jpg'))]
               for c in classes])

    
    return list(dataset)



class Food101Dataset(mango.SplitDataset):

    path = mango.Param(str)
    seed = mango.Param(int)

    def __init__(self, path, classes, seed=42):
        super().__init__(path=path, seed=seed)
        self.classes = classes
        
    def build(self):
        np.random.seed(42)
        filenames = load_filenames(self.path, self.classes)
        self.train_fn, self.test_fn = train_test_split(filenames, test_size=512)
        
    def train(self):
        return self.train_fn
    
    def test(self):
        return self.test_fn
    
    def transform_train(self, data):
        list_of_dicts = _pool.starmap(self.get, [(image_fname, klass, 'train') for klass, image_fname in data])
        images = list(pluck('images', list_of_dicts))
        labels = list(pluck('labels', list_of_dicts))
        
        return {'images': np.array(images), 'labels': np.array(labels)}
    
    def transform_test(self, data):
        list_of_dicts = [self.get(image_fname, klass, 'test') for klass, image_fname in data] 
        images = list(pluck('images', list_of_dicts))
        labels = list(pluck('labels', list_of_dicts))
        
        return {'images': np.array(images), 'labels': np.array(labels)}
        
        

    def get(self, image_fname, klass, mode):
        img = Image.open(image_fname)
        
        if mode == 'test':
            return {'images': self._eval_image(img), 'labels': klass}
        elif mode == 'train':
            return {'images': self._noise_image(img), 'labels': hash(klass)} 
        else:
            raise ValueError('Value not good')

    def _noise_image(self, img):
        transform = transforms.Compose(
            [   
                transforms.RandomRotation(60),
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor()
            ])
        return transform(img).numpy()
    
    def _eval_image(self, img):
        transform = transforms.Compose(
            [   
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        return transform(img).numpy()
    
_pool = multiprocessing.Pool()
