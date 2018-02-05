import itertools
from toolz import groupby, take, concat
import numpy as np

# How do we randomize the pairwise transformation?

class PairwiseTransform:

    def __init__(self, num_positive, num_negative):
        self.num_positive = num_positive
        self.num_negative = num_negative

    def transform(self, X, y):
        positives = []
        negatives = []
        items = groupby(lambda i: y[i], range(len(y))).items()

        for label, group_ix in items:

            possible_positives = len(group_ix) ** 2
            if self.num_positive >= possible_positives:
                raise ValueError('Not enough combinations for positive examples')

            group = X[group_ix]
            positives.extend(take(self.num_positive, itertools.product(group_ix,np.random.permutation(group_ix))))

            other_candidates_ix = np.concatenate([group_ix for label_neg, group_ix in items if label_neg != label])

            possible_negatives = len(other_candidates_ix) * len(group_ix)
            if self.num_negative >= possible_negatives:
                raise ValueError('Not enough combinations for negative examples')

            negatives.extend(take(self.num_negative, itertools.product(group_ix, np.random.permutation(other_candidates_ix))))

        all_pairs = positives + negatives
        all_output = [1] * len(positives) + [0] * len(negatives)

        return X.take(all_pairs, axis=0), all_output
