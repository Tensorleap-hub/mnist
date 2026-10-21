from code_loader.default_metrics import categorical_crossentropy
from code_loader.visualizers.default_visualizers import default_image_visualizer

from mnist.data.preprocess import preprocess_func
from mnist.utils import *
from mnist.config import CONFIG
from code_loader.inner_leap_binder.leapbinder_decorators import *
from code_loader.contract.datasetclasses import DatasetMetadataType, MetricDirection, SamplePreprocessResponse
from numpy.typing import NDArray
from typing import Optional
import json


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    train_X, val_X, train_Y, val_Y = preprocess_func(CONFIG['local_file_path'])
    train_X, train_Y = train_X[:2000], train_Y[:2000]
    val_X, val_Y = val_X[:1000], val_Y[:1000]

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=len(train_X), data={'images': train_X, 'labels': train_Y}, state=DataStateType.training)
    val = PreprocessResponse(length=len(val_X), data={'images': val_X, 'labels': val_Y}, state=DataStateType.validation)
    leap_binder.cache_container["classes_avg_images"] = calc_classes_centroid(train_X, train_Y)
    response = [train, val]
    return response


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image', channel_dim=-1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['images'][idx].astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_one_hot_digit')
def metadata_one_hot_digit(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)

    res = {
        'label': str(metadata_label(digit_int)),
        'even_odd': metadata_even_odd(digit_int),
        'circle': metadata_circle(digit_int)
    }
    return res


@tensorleap_metadata('euclidean_diff_from_class_centroid')
def metadata_euclidean_distance_from_class_centroid(idx: int,
                                                    preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    ### calculate euclidean distance from the average image of the specific class
    sample_input = preprocess.data['images'][idx]
    label = preprocess.data['labels'][idx]
    label = str(np.argmax(label))
    class_average_image = leap_binder.cache_container["classes_avg_images"][label]
    return np.linalg.norm(class_average_image - sample_input)


@tensorleap_custom_visualizer('horizontal_bar_classes', LeapHorizontalBar.type)
def combined_bar(data: NDArray[float], gt:NDArray[float]) -> LeapHorizontalBar:
    return LeapHorizontalBar(np.squeeze(data), gt=np.squeeze(gt), labels=CONFIG['LABELS'])


@tensorleap_custom_metric('metrics',direction=MetricDirection.Downward)
def metrics(output_pred: NDArray[float]) -> Dict[str, NDArray[Union[float, int]]]:
    prob = output_pred.max(axis=-1)
    pred_idx = output_pred.argmax(axis=-1)
    metrics_dict = {'prob': prob,
                    'prd_idx': pred_idx}
    return metrics_dict



IMMENSE_BYTES = 40_000  # > Elasticsearch keyword max term length (32766)


# Two ways Elasticsearch rejects a metrics doc, which the engine then silently drops.
# Validation idx % 4 == 0 -> oversized string; idx % 4 == 1 -> 'NaT' in a date-detected field.
# Validation idx % 4 in (2, 3) and all training samples are clean controls.
def _is_oversized_sample(idx: int, preprocess: PreprocessResponse) -> bool:
    return preprocess.state == DataStateType.validation and idx % 4 == 0


def _is_nat_sample(idx: int, preprocess: PreprocessResponse) -> bool:
    return preprocess.state == DataStateType.validation and idx % 4 == 1


@tensorleap_metadata('evaluation_table_json', DatasetMetadataType.string)
def evaluation_table_json(idx: int, preprocess: PreprocessResponse) -> str:
    """Mimics the Blue River integration: a JSON dump of the whole evaluation-table row.
    Rows selected by _is_oversized_sample exceed the keyword limit; all others are small."""
    digit = int(preprocess.data['labels'][idx].argmax())
    if _is_oversized_sample(idx, preprocess):
        payload = {f'col_{i}': f'{digit}-{idx}-' + 'x' * 20 for i in range(IMMENSE_BYTES // 30)}
    else:
        payload = {'digit': digit, 'idx': idx}
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


@tensorleap_metadata('annotation_updated_at', DatasetMetadataType.string)
def annotation_updated_at(idx: int, preprocess: PreprocessResponse) -> str:
    """Mimics Blue River's timestamp columns: pandas NaT is serialized as the string 'NaT'.
    Rows selected by _is_nat_sample get 'NaT'; all others carry a real ISO timestamp."""
    if _is_nat_sample(idx, preprocess):
        return 'NaT'
    return f'2026-08-{idx % 28 + 1:02d}T10:00:00'


@tensorleap_custom_loss('categorical_crossentropy_loss')
def categorical_crossentropy_loss(ground_truth: np.array, prediction: np.array) -> np.array:
    return categorical_crossentropy(ground_truth, prediction)


@tensorleap_custom_visualizer('default_image_visualizer', LeapDataType.Image)
def image_visualizer(data: np.float32):
    return default_image_visualizer(data)




if __name__ == '__main__':
    leap_binder.check()
