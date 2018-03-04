import os
import glob
import numpy as np

from toolz import concat
from skimage import io, transform
from torch.utils.data import Dataset
from food.itemz import Collection
from ..transformers.pairwise import PairwiseTransform


def load_filenames(path):
    '''
    Create a dataset where to a picture we have an associated label
    '''

    classes = os.listdir(os.path.join(path, 'images'))
    dataset = concat([[(c, img) for img in glob.glob(os.path.join(path, 'images', c, '*.jpg'))]
               for c in classes])

    return list(dataset)

def make_dataset(path):
    c = Collection.create((load_filenames, path))
    c_train, c_test = (c.groupby(0)  # Group by class
                       .evolve(1, lambda x: list(pluck(1, x))) # Remove class from tuple
                       .combinations(replacement=True) # Generate combinations for classes
                       .starmap(generate_matching_pairs) # Generate one-hot
                       .flatten()
                       .persist()
                       .filter(random_drop(0.5)) # Subsample the negative examples (too many)
                       .random_split(0.70))


def generate_matching_pairs(group1, group2):
    same_key = group1[0] == group2[0]
    return [(same_key, k1, k2) for k1 in group1[1] for k2 in group2[1]]

@curry
def random_drop(proportion, seed, data):
    random.seed(seed)
    if random.random() < proportion and data[0] is False:
        return False
    else:
        return True

class Food101PairwiseDataset(Dataset):

    def __init__(self, path, test_proportion=0.1):
        self.path = path
        train, test = (Collection.create((load_filenames, path))
                                 .random_split(test_proportion, 1))

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
        img = io.imread(fname)
        img = transform.resize(img, (224, 224))
        return img

    def test(self):
        self.data = self._test_data.compute()

    def train(self):
        self.data = self._train_data.compute()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        match, imga, imgb = self.data[idx]

        imga = self._process_image(imga)
        imgb = self._process_image(imgb)

        sample = {'pair': np.array([img_a, img_b]), 'match': match}

        return sample
