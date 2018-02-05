from food.transformers.pairwise import PairwiseTransform
import numpy as np


def test_siamese_net():


    clf = SiameseNet()
    # For a siamese network you need to provide a pair, and give 0, 1
    clf.fit(X, y)


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
