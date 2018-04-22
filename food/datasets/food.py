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


def generate_matching_pairs(group1, group2):
    key1, items1 = group1
    key2, items2 = group2
    same_key = key1 == key2

    if same_key:
        k = 10
    else:
        k = 2

    random.shuffle(items1)
    random.shuffle(items2)
    
    items1 = items1[:k]
    items2 = items2[:k]
    
    return [(same_key, k1, k2) for k1 in items1 for k2 in items2]

class Food101PairwiseDataset(Dataset):

    def __init__(self, path, test_proportion=0.1, resnet_format=False):
        self.path = path
        self.resnet_format = resnet_format
        train, test = (Collection.create((load_filenames, path))
                                 .random_split(test_proportion, 1))
        train = train.persist()
        test = test.persist()

        pipe = (Collection.pipeline('file_list')
               .groupby(0)  # Group by class
               .evolve(1, lambda x: list(pluck(1, x))) # Remove class from tuple
               .combinations(replacement=True) # Generate combinations for classes
               .starmap(generate_matching_pairs) # Generate one-shot triplets
               .flatten())

        self._train_data = pipe.pump({'file_list': train})
        self._test_data = pipe.pump({'file_list': test})

        self.train()

    def _process_image(self, fname):
        img = Image.open(fname)
        # img = transform.resize(img, (224, 224))
        #if self.resnet_format:
        #    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
        #    img = img.astype('float32') / 255
        
        return self._noise_image(img)

    
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
    def test(self):
        self.data = self._test_data.compute()

    def train(self):
        random.seed()
        self.data = self._train_data.compute()
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        match, imga, imgb = self.data[idx]

        imga = self._process_image(imga)
        imgb = self._process_image(imgb)

        sample = {'pair': np.array([imga, imgb]), 'match': match}

        return sample

from sklearn.model_selection import train_test_split


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
        list_of_dicts = [self.get(image_fname, klass, 'train') for klass, image_fname in data]
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