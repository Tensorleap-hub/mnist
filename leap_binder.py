from code_loader.default_metrics import categorical_crossentropy
from code_loader.visualizers.default_visualizers import default_image_visualizer

from mnist.data.preprocess import preprocess_func
from mnist.utils import *
from mnist.config import CONFIG
from code_loader.inner_leap_binder.leapbinder_decorators import *
from numpy.typing import NDArray

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image')
def input_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['images'][int(idx)].astype('float32')


@tensorleap_instances_masks_encoder('image')
def instance_mask_encoder(idx: str, preprocess: PreprocessResponse, instance_idx) -> ElementInstance:
    inp = input_encoder(idx, preprocess)
    gt = [(0, 0, 16, 16, 0), (7, 7, 13, 13, 1)]
    label = gt[instance_idx]

    mask = np.zeros_like(inp)
    x, y, w, h, label_id = label
    # img_width, img_height = mask.shape[1], mask.shape[2]
    # x, y, w, h = round(x * img_width), round(y * img_height), round(w * img_width), round(h * img_height)
    mask[y:y+h, x:x+w, :] = 1

    extra_metadata = {"instance_size": int(np.sum(mask)), "namee": f"aa_{instance_idx}"}
    element_instances = ElementInstance(f"{label_id}", mask, extra_metadata)

    return element_instances

@tensorleap_instances_length_encoder('image')
def instances_length_encoder(idx: str, preprocess: PreprocessResponse) -> int:
    return 2

@tensorleap_element_instance_preprocess(instances_length_encoder, instance_mask_encoder)
def preprocess_func_leap() -> List[PreprocessResponse]:
    train_X, val_X, train_Y, val_Y = preprocess_func(CONFIG['local_file_path'])

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(sample_ids=[str(i) for i in range(int(len(train_X)/1000))], data={'images': train_X, 'labels': train_Y})
    val = PreprocessResponse(sample_ids=[str(i) for i in range(int(len(val_X)/1000))], data={'images': val_X, 'labels': val_Y})
    leap_binder.cache_container["classes_avg_images"] = calc_classes_centroid(train_X, train_Y)
    response = [train, val]
    return response




# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][int(idx)].astype('float32')


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: str, preprocess: PreprocessResponse) -> int:
    return int(idx)


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_one_hot_digit')
def metadata_one_hot_digit(idx: str, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
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
def metadata_euclidean_distance_from_class_centroid(idx: str,
                                                    preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    ### calculate euclidean distance from the average image of the specific class
    sample_input = preprocess.data['images'][int(idx)]
    label = preprocess.data['labels'][int(idx)]
    label = str(np.argmax(label))
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


@tensorleap_custom_visualizer('image_visualizer_original', LeapDataType.Image)
def image_visualizer_original(image: np.ndarray, sample_preprocess_response: SamplePreprocessResponse):
    id = sample_preprocess_response.sample_ids[0]
    sample_id = sample_preprocess_response.preprocess_response.instance_to_sample_ids_mappings[str(id)]
    orig_image = input_encoder(sample_id, sample_preprocess_response.preprocess_response)
    return default_image_visualizer(np.array([orig_image]))


if __name__ == '__main__':
    leap_binder.check()
