from code_loader.default_metrics import categorical_crossentropy
from code_loader.visualizers.default_visualizers import default_image_visualizer

from mnist.data.preprocess import (preprocess_func, read_sample_image, read_sample_label,
                                    compute_class_centroids)
from mnist.utils import *
from mnist.config import CONFIG
from code_loader.inner_leap_binder.leapbinder_decorators import *
from keras.utils import to_categorical
from numpy.typing import NDArray


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    # Parquet-backed storage: each split is many small Parquet files. `preprocess_func`
    # ensures the files exist and returns, per split, the flat list of string sample_ids
    # ("{file_idx}_{row}") plus the ordered list of file paths (in `data`). The encoders
    # resolve a sample_id -> (file, row) and read that single sample.
    splits = preprocess_func(CONFIG['local_file_path'])
    train_ids, train_data = splits['train']
    val_ids, val_data = splits['val']

    train = PreprocessResponse(sample_ids=train_ids, data=train_data, sample_id_type=str)
    val = PreprocessResponse(sample_ids=val_ids, data=val_data, sample_id_type=str)
    leap_binder.cache_container["classes_avg_images"] = compute_class_centroids()
    response = [train, val]
    return response


# Input encoder reads the image for `sample_id` from its Parquet file (LRU-cached).
# Returns a (28, 28, 1) float32 image normalized to [0, 1].
@tensorleap_input_encoder('image')
def input_encoder(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    return read_sample_image(sample_id, preprocess)


# Ground truth encoder reads the label for `sample_id` and returns a one-hot float32 vector.
@tensorleap_gt_encoder('classes')
def gt_encoder(sample_id: str, preprocessing: PreprocessResponse) -> np.ndarray:
    label = read_sample_label(sample_id, preprocessing)
    return to_categorical(label, num_classes=len(CONFIG['LABELS'])).astype('float32')


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata records the sample's id.
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(sample_id: str, preprocess: PreprocessResponse) -> str:
    return sample_id


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_one_hot_digit')
def metadata_one_hot_digit(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
    one_hot_digit = gt_encoder(sample_id, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)

    res = {
        'label': str(metadata_label(digit_int)),
        'even_odd': metadata_even_odd(digit_int),
        'circle': metadata_circle(digit_int)
    }
    return res


@tensorleap_metadata('euclidean_diff_from_class_centroid')
def metadata_euclidean_distance_from_class_centroid(sample_id: str,
                                                    preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    ### calculate euclidean distance from the average image of the specific class
    sample_input = read_sample_image(sample_id, preprocess)
    label = str(read_sample_label(sample_id, preprocess))
    class_average_image = leap_binder.cache_container["classes_avg_images"][label]
    return np.linalg.norm(class_average_image - sample_input)


@tensorleap_custom_visualizer('horizontal_bar_classes', LeapHorizontalBar.type)
def combined_bar(data: NDArray[float], gt:NDArray[float]) -> LeapHorizontalBar:
    return LeapHorizontalBar(np.squeeze(data), gt=np.squeeze(gt), labels=CONFIG['LABELS'])


@tensorleap_custom_metric('metrics')
def metrics(output_pred: NDArray[float]) -> Dict[str, NDArray[Union[float, int]]]:
    prob = output_pred.max(axis=-1)
    pred_idx = output_pred.argmax(axis=-1)
    metrics_dict = {'prob': prob,
                    'prd_idx': pred_idx}
    return metrics_dict


@tensorleap_custom_loss('categorical_crossentropy_loss')
def categorical_crossentropy_loss(ground_truth: np.array, prediction: np.array) -> np.array:
    return categorical_crossentropy(ground_truth, prediction)


@tensorleap_custom_visualizer('default_image_visualizer', LeapDataType.Image)
def image_visualizer(data: np.float32):
    return default_image_visualizer(data)




if __name__ == '__main__':
    leap_binder.check()
