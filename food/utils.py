import numpy as np
from skimage import transform

def channels_last_to_first(image, axis_offset=0):
    '''Transform image from shape (width, height, channels) to (channels, width, height)'''
    converted = np.swapaxes(np.swapaxes(image, axis_offset + 1, axis_offset + 2), axis_offset + 0, axis_offset + 1)
    return converted

def to_resnet(img):
    img = transform.resize(img, (224, 224))
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    return img