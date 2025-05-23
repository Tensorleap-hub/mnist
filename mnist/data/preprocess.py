from typing import Union, Iterable
import numpy as np
from numpy import ndarray
from keras.datasets import mnist
from keras.utils import to_categorical
import os

from mnist.config import CONFIG

def get_full_path(path):
    if os.path.isabs(path):
        return path
    elif 'IS_CLOUD' in os.environ and os.path.exists("/nfs"): # SaaS env
        return os.path.join("/nfs", path)
    elif 'GENERIC_HOST_PATH' in os.environ: # OnPrem
        return os.path.join(os.environ['GENERIC_HOST_PATH'], path)
    else:
        return os.path.join(os.path.expanduser('~'), 'tensorleap/data', CONFIG['local_file_path'])


def preprocess_func(local_file_path) -> Union[ndarray, Iterable, int, float, tuple, dict]:
    # Check if the data directory exists, and create it if not
    if local_file_path is None:
        local_file_path = CONFIG['local_file_path']
    local_file_path = get_full_path(local_file_path)
    if not os.path.exists(local_file_path):
        os.makedirs(local_file_path)
    data_file = os.path.join(local_file_path, 'mnist.npz')

    if not os.path.exists(data_file):
        # Data file doesn't exist, download and save it
        (train_X, train_Y), (val_X, val_Y) = mnist.load_data(data_file)
    else:
        data = np.load(data_file)
        train_X, val_X, train_Y, val_Y = data['x_train'], data['x_test'], data['y_train'], data['y_test']
        
    train_X = np.expand_dims(train_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    train_X = train_X / 255  # Normalize to [0,1]
    train_Y = to_categorical(train_Y)  # Hot Vector

    val_X = np.expand_dims(val_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    val_X = val_X / 255  # Normalize to [0,1]
    val_Y = to_categorical(val_Y)  # Hot Vector
            
    return train_X, val_X, train_Y, val_Y