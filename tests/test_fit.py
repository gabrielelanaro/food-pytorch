from food.transformers.pairwise import PairwiseTransform
from food.training.trainer import Trainer

import numpy as np


def make_dataset(n_images, n_classes):
    random_images = np.random.randint(0, 256, size=(n_images, 3, 224, 224))
    random_classes = np.random.randint(0, n_classes, size=n_images)

    random_images = random_images.astype('float32') / 256
    return random_images, random_classes

def test_siamese_net():
    n_images = 100
    n_classes = 4

    X, y = make_dataset(n_images, n_classes)

    tf = PairwiseTransform(3, 4)

    clf = Trainer()
    # For a siamese network you need to provide a pair, and give 0, 1
    X_tf, y_tf = tf.transform(X, y)

    print(X_tf.shape)
    clf.fit_batch(X_tf, y_tf)


def test_pairwise():
    num_classes = 3
    num_positive = 10
    num_negative = 10
    num_samples = 100
    feature_size = 20

    # 100 samples
    X_src = np.zeros((num_samples, feature_size))

    # 100 labels
    y_src = np.random.randint(low=0, high=num_classes, size=(num_samples))

    tf = PairwiseTransform(num_positive, num_negative)

    X_dst, y_dst = tf.transform(X_src, y_src)

    assert len(X_dst) == len(y_dst) == num_classes * (num_positive + num_negative)
    assert X_dst.shape == (num_classes * (num_positive + num_negative), 2, feature_size)
