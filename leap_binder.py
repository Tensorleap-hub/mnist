import os

from code_loader.default_metrics import categorical_crossentropy
from code_loader.visualizers.default_visualizers import default_image_visualizer

from mnist.data.preprocess import (preprocess_func, read_sample_image, read_sample_label,
                                    compute_class_centroids, split_into_groups)
from mnist.utils import *
from mnist.config import CONFIG
from code_loader.inner_leap_binder.leapbinder_decorators import *
from keras.utils import to_categorical
from numpy.typing import NDArray


def _grouped_settings():
    """Whether to emit grouped (List[List[sample_id]]) preprocess output, and into how
    many groups. Reads `grouped:` from project_config.yaml; env vars MNIST_GROUPED /
    MNIST_NUM_GROUPS override it so flat and grouped can be run without editing files."""
    cfg = CONFIG.get('grouped', {}) or {}
    env = os.environ.get('MNIST_GROUPED')
    enabled = (env == '1') if env is not None else bool(cfg.get('enabled', False))
    num_groups = int(os.environ.get('MNIST_NUM_GROUPS', cfg.get('num_groups', 4)))
    return enabled, num_groups


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    # Parquet-backed storage: each split is many small Parquet files. `preprocess_func`
    # ensures the files exist and returns, per split, the flat list of string sample_ids
    # ("{file_idx}_{row}") plus the ordered list of file paths (in `data`). The encoders
    # resolve a sample_id -> (file, row) and read that single sample.
    splits = preprocess_func(CONFIG['local_file_path'])
    train_ids, train_data = splits['train']
    val_ids, val_data = splits['val']

    # List-of-inputs path: cut each split's flat ids into N disjoint groups. The encoders
    # then receive a whole group per call and return a (B, *) batch (see below).
    grouped, num_groups = _grouped_settings()
    if grouped:
        train_ids = split_into_groups(train_ids, num_groups)
        val_ids = split_into_groups(val_ids, num_groups)

    train = PreprocessResponse(sample_ids=train_ids, data=train_data, sample_id_type=str)
    val = PreprocessResponse(sample_ids=val_ids, data=val_data, sample_id_type=str)
    leap_binder.cache_container["classes_avg_images"] = compute_class_centroids()
    response = [train, val]
    return response


# Input encoder reads the image for `sample_id` from its Parquet file (LRU-cached).
# Flat call (str id) -> a single (28, 28, 1) float32 image normalized to [0, 1].
# Grouped call (list of ids) -> a list of per-sample (28, 28, 1) arrays, one per id.
@tensorleap_input_encoder('image')
def input_encoder(sample_id: Union[str, List[str]], preprocess: PreprocessResponse) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(sample_id, list):
        return [read_sample_image(sid, preprocess) for sid in sample_id]
    return read_sample_image(sample_id, preprocess)


# Per-sample core; reused by the flat and grouped branches. Kept undecorated so the
# grouped branch can call it per id without tripping the decorator's id-vs-group check.
def _encode_gt_single(sample_id: str, preprocessing: PreprocessResponse) -> np.ndarray:
    label = read_sample_label(sample_id, preprocessing)
    return to_categorical(label, num_classes=len(CONFIG['LABELS'])).astype('float32')


# Ground truth encoder reads the label for `sample_id` and returns a one-hot float32 vector.
# Grouped call -> a list of per-sample (num_classes,) one-hot arrays, one per id.
@tensorleap_gt_encoder('classes')
def gt_encoder(sample_id: Union[str, List[str]], preprocessing: PreprocessResponse) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(sample_id, list):
        return [_encode_gt_single(sid, preprocessing) for sid in sample_id]
    return _encode_gt_single(sample_id, preprocessing)


# Metadata functions allow to add extra data for a later use in analysis. Each is
# group-aware: a flat call (str id) returns one value; a grouped call (list of ids)
# returns a per-sample list of the same values, one entry per id in the group.

# This metadata records the sample's id.
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(sample_id: Union[str, List[str]], preprocess: PreprocessResponse) -> Union[str, List[str]]:
    if isinstance(sample_id, list):
        return list(sample_id)
    return sample_id


# Per-sample core for the digit-label metadata (undecorated; see _encode_gt_single).
def _one_hot_digit_single(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
    one_hot_digit = _encode_gt_single(sample_id, preprocess)
    digit_int = int(one_hot_digit.argmax())
    return {
        'label': str(metadata_label(digit_int)),
        'even_odd': metadata_even_odd(digit_int),
        'circle': metadata_circle(digit_int)
    }


# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_one_hot_digit')
def metadata_one_hot_digit(sample_id: Union[str, List[str]], preprocess: PreprocessResponse):
    if isinstance(sample_id, list):
        return [_one_hot_digit_single(sid, preprocess) for sid in sample_id]
    return _one_hot_digit_single(sample_id, preprocess)


# Per-sample core for the class-centroid distance metadata (undecorated; see above).
def _euclidean_single(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    ### calculate euclidean distance from the average image of the specific class
    sample_input = read_sample_image(sample_id, preprocess)
    label = str(read_sample_label(sample_id, preprocess))
    class_average_image = leap_binder.cache_container["classes_avg_images"][label]
    return np.linalg.norm(class_average_image - sample_input)


@tensorleap_metadata('euclidean_diff_from_class_centroid')
def metadata_euclidean_distance_from_class_centroid(sample_id: Union[str, List[str]],
                                                    preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    if isinstance(sample_id, list):
        return [_euclidean_single(sid, preprocess) for sid in sample_id]
    return _euclidean_single(sample_id, preprocess)


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
