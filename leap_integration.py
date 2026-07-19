import os
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize

from mnist.config import CONFIG
from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         combined_bar, metrics, image_visualizer, categorical_crossentropy_loss,
                         metadata_sample_index, metadata_one_hot_digit, metadata_euclidean_distance_from_class_centroid,
                         _grouped_settings)
import tensorflow as tf
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test

prediction_type1 = PredictionTypeHandler('classes', CONFIG['LABELS'])

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/model.h5'
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return cnn


@tensorleap_integration_test()
def check_custom_test_mapping(idx, subset):
    image = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)

    # Grouped datasets hand the encoders a whole group and get back a per-sample list;
    # index one sample to wire it through the model. This indexing is only legal on a
    # grouped dataset (the decorator rejects it when flat), so branch on the toggle.
    if _grouped_settings()[0]:
        image = image[0]
        gt = gt[0]

    cnn = load_model()
    y_pred = cnn([image])
    both_vis = combined_bar(y_pred, gt)
    img_vis = image_visualizer(image)

    visualize(img_vis)

    metric_result = metrics(y_pred)

    loss_ret = categorical_crossentropy_loss(gt, y_pred)

    m1 = metadata_sample_index(idx, subset)
    print(m1)
    m2 = metadata_one_hot_digit(idx, subset)
    m3 = metadata_euclidean_distance_from_class_centroid(idx, subset)

    # here the user can return whatever he wants


if __name__ == '__main__':
    train_subset = preprocess_func_leap()[0]
    check_custom_test_mapping(train_subset.sample_ids[0], train_subset)












