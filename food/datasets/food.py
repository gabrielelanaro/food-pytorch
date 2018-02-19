import os
import glob
import numpy as np

from toolz import concat
from skimage import io, transform
from torch.utils.data import Dataset

from ..transformers.pairwise import PairwiseTransform


def load_filenames(path):
    '''
    Create a dataset where to a picture we have an associated label
    '''

    classes = os.listdir(os.path.join(path, 'images'))
    dataset = concat([[(img, c) for img in glob.glob(os.path.join(path, 'images', c, '*.jpg'))]
               for c in classes])

    return list(dataset)



class Food101PairwiseDataset(Dataset):

    def __init__(self, path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        num_pos = 3
        num_neg = 3

        self.path = path
        self.transform = transform
        X, y = zip(*load_filenames(self.path))

        self.X = np.array(X)
        self.y = np.array(y)
        self.tf = PairwiseTransform(num_pos, num_neg)
        self.X_tf, self.y_tf = self.tf.transform(self.X, self.y)

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        fname_a, fname_b = self.X_tf[idx]
        out = self.y_tf[idx]

        img_a, img_b = io.imread(fname_a), io.imread(fname_b)
        img_a = transform.resize(img_a, (224, 224))
        img_b = transform.resize(img_b, (224, 224))

        sample = {'pair': np.array([img_a, img_b]), 'match': out}

        return sample
