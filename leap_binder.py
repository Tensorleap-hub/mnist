from typing import List, Union, Dict
import os

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapHorizontalBar

from mnist.data.preprocess import preprocess_func
from mnist.utils import *
from mnist.config import CONFIG


def preprocess_func_leap() -> List[PreprocessResponse]:
    root_path = os.path.expanduser("~")
    local_file_path = os.path.join(root_path, CONFIG['local_file_path'])
    data = preprocess_func(local_file_path)
    train_X, val_X, train_Y, val_Y = data['train_X'], data['val_X'], data['train_Y'], data['val_Y']

    # Get random indices for downsampling
    indices = np.random.choice(len(train_X), CONFIG['train_size'], replace=False)

    # Downsample the datasets using the random indices
    train_X, train_Y = train_X[indices], train_Y[indices]

    # Repeat for the validation set if needed
    indices_val = np.random.choice(len(val_X), CONFIG['val_size'], replace=False)
    val_X, val_Y = val_X[indices_val], val_Y[indices_val]

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=len(train_X), data={'images': train_X, 'labels': train_Y})
    val = PreprocessResponse(length=len(val_X), data={'images': val_X, 'labels': val_Y})
    leap_binder.cache_container["classes_avg_images"] = calc_classes_centroid(train_X, train_Y)
    response = [train, val]
    return response


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['images'][idx].astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


def metadata_one_hot_digit(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)

    res = {
        'label': metadata_label(digit_int),
        'even_odd': metadata_even_odd(digit_int),
        'circle': metadata_circle(digit_int)
    }
    return res


def metadata_euclidean_distance_from_class_centroid(idx: int,
                                                    preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    ### calculate euclidean distance from the average image of the specific class
    sample_input = preprocess.data['images'][idx]
    label = preprocess.data['labels'][idx]
    label = str(np.argmax(label))
    class_average_image = leap_binder.cache_container["classes_avg_images"][label]
    return np.linalg.norm(class_average_image - sample_input)


def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    return LeapHorizontalBar(data, CONFIG['LABELS'])


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func_leap)
leap_binder.set_input(function=input_encoder, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
leap_binder.set_metadata(function=metadata_sample_index, name='metadata_sample_index')
leap_binder.set_metadata(function=metadata_one_hot_digit, name='metadata_one_hot_digit')
leap_binder.set_metadata(function=metadata_euclidean_distance_from_class_centroid,
                         name='euclidean_diff_from_class_centroid')
leap_binder.add_prediction(name='classes', labels=CONFIG['LABELS'])
leap_binder.set_visualizer(name='horizontal_bar_classes', function=bar_visualizer,
                           visualizer_type=LeapHorizontalBar.type)

if __name__ == '__main__':
    leap_binder.check()
