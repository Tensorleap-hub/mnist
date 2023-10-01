import numpy as np
from mnist.config import CONFIG


def metadata_label(digit_int) -> int:
    return digit_int


def metadata_even_odd(digit_int) -> str:
    if digit_int % 2 == 0:
        return "even"
    else:
        return "odd"


def metadata_circle(digit_int) -> str:
    if digit_int in [0, 6, 8, 9]:
        return 'yes'
    else:
        return 'no'


def calc_classes_centroid(data_X: np.ndarray, data_Y: np.ndarray) -> dict:
    avg_images_dict = {}
    # calculate average image on the pixels.
    # returns a dictionary: key: class, values: images 28x28
    for label in CONFIG['LABELS']:
        inputs_label = data_X[np.equal(np.argmax(data_Y, axis=1), int(label))]
        avg_images_dict[label] = np.mean(inputs_label, axis=0)
    return avg_images_dict
