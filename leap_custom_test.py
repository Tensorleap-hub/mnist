from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         combined_bar, leap_binder, metrics, image_visualizer)
import tensorflow as tf
import os
import numpy as np
from code_loader.helpers import visualize


def check_custom_test():
    check_generic = True
    plot_vis = True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/model.h5'
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    responses = preprocess_func_leap()
    for subset in responses:  # train, val
        for idx in range(3):  # analyze first 3 images
            # get input and gt
            image = input_encoder(idx, subset)
            img_vis = image_visualizer(image)
            gt = gt_encoder(idx, subset)

            # add batch to input & gt
            concat = np.expand_dims(image, axis=0)

            # infer model
            y_pred = cnn([concat])

            # get inputs & outputs (no batch)
            both_vis = combined_bar(y_pred.numpy(), gt)

            # plot inputs & outputs
            if plot_vis:
                visualize(both_vis)
                visualize(img_vis)

            # print metrics
            metric_result = metrics(y_pred.numpy())
            print(metric_result)

            # print metadata
            for metadata_handler in leap_binder.setup_container.metadata:
                curr_metadata = metadata_handler.function(idx, subset)
                print(f"Metadata {metadata_handler.name}: {curr_metadata}")

    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
