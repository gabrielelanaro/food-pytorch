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

def load_filenames(path):
    '''
    Create a dataset where to a picture we have an associated label
    '''

    classes = os.listdir(os.path.join(path, 'images'))
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

    
    
class Food101Dataset:

    path: str

    def __init__(self, path):
        self.path = path
        
    def load(self):
        self.filenames = load_filenames(self.path)

    def __getitem__(self, idx):
        klass, image_fname = self.filenames[idx]
        img = self._process_image(image_fname)
        return {'images': img, 'labels': hash(klass)}
        
    def __len__(self):
        return len(self.filenames)
    
    def _process_image(self, fname):
        img = Image.open(fname)
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