import numpy as np

def channels_last_to_first(image, axis_offset=0):
    '''Transform image from shape (width, height, channels) to (channels, width, height)'''
    converted = np.swapaxes(np.swapaxes(image, axis_offset + 1, axis_offset + 2), axis_offset + 0, axis_offset + 1)
    return converted
