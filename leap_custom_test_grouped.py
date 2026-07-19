"""List-of-inputs ("grouped") validation for the MNIST integration.

Runs the same integration as the flat baseline, but with `preprocess_func_leap`
emitting nested sample_ids (List[List[sample_id]]) — each split cut into N disjoint,
near-equal groups. Validates three layers:

  1. leap_binder.check()               -> grouped registration + shape/type validation
  2. the group-aware encoders directly -> input (B,*), gt (B,*), metadata per-sample lists
  3. LeapLoader.get_samples(...)        -> the real engine fetch-by-group path (order preserved)

Grouping is forced on via MNIST_GROUPED=1 (set before importing leap_binder) so this
script is self-contained and leaves the flat baseline untouched.
"""
import os

os.environ["MNIST_GROUPED"] = "1"  # must be set before leap_binder / preprocess run
os.environ.setdefault("MNIST_NUM_GROUPS", "4")

import numpy as np
import tensorflow as tf

from code_loader.contract.enums import DataStateEnum
from code_loader.leaploader import LeapLoader

from leap_binder import (
    preprocess_func_leap, input_encoder, gt_encoder, leap_binder,
    metadata_sample_index, metadata_one_hot_digit,
    metadata_euclidean_distance_from_class_centroid,
)
from mnist.config import CONFIG

NUM_CLASSES = len(CONFIG["LABELS"])


def _check_grouped_preprocess():
    print("\n[1] leap_binder.check() on grouped preprocess ...")
    leap_binder.check()

    responses = preprocess_func_leap()
    for split_name, subset in zip(("train", "val"), responses):
        ids = subset.sample_ids
        assert isinstance(ids, list) and all(isinstance(g, list) for g in ids), \
            f"{split_name}: expected nested List[List[sample_id]], got {type(ids)}"
        flat = [sid for g in ids for sid in g]
        assert len(flat) == len(set(flat)), f"{split_name}: groups overlap"
        sizes = [len(g) for g in ids]
        assert max(sizes) - min(sizes) <= 1, f"{split_name}: groups not near-equal: {sizes}"
        print(f"    {split_name}: {len(ids)} groups, sizes={sizes}, total={len(flat)} samples")
    return responses


def _check_encoders_on_group(subset):
    print("\n[2] group-aware encoders on one full group ...")
    group = subset.sample_ids[0]
    b = len(group)

    imgs = input_encoder(group, subset)
    gts = gt_encoder(group, subset)
    # grouped encoders return a list of per-sample arrays (one entry per id in the group)
    assert isinstance(imgs, list) and len(imgs) == b, f"expected list of {b} images, got {type(imgs)}"
    assert isinstance(gts, list) and len(gts) == b, f"expected list of {b} gts, got {type(gts)}"
    assert imgs[0].shape == (28, 28, 1), f"per-sample image shape {imgs[0].shape} != (28, 28, 1)"
    assert gts[0].shape == (NUM_CLASSES,), f"per-sample gt shape {gts[0].shape} != {(NUM_CLASSES,)}"

    # each grouped row must equal encoding that id on its own (as a singleton group —
    # a single id against a grouped dataset is (correctly) rejected by the contract).
    # Spot-check the first few rows; the full group can be large.
    for i, sid in enumerate(group[:5]):
        assert np.array_equal(imgs[i], input_encoder([sid], subset)[0]), f"input row {i} mismatch"
        assert np.array_equal(gts[i], gt_encoder([sid], subset)[0]), f"gt row {i} mismatch"

    idx_md = metadata_sample_index(group, subset)
    digit_md = metadata_one_hot_digit(group, subset)
    dist_md = metadata_euclidean_distance_from_class_centroid(group, subset)
    assert idx_md == list(group), "sample_index metadata should echo the group ids in order"
    assert len(digit_md) == b and all(isinstance(d, dict) for d in digit_md)
    assert len(dist_md) == b
    print(f"    group of {b}: input list[{b}] of {imgs[0].shape}, "
          f"gt list[{b}] of {gts[0].shape}, metadata lists len={b} OK")
    return group


def _check_get_samples(group):
    print("\n[3] LeapLoader.get_samples on the same group (engine fetch-by-group) ...")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    loader = LeapLoader(dir_path, "leap_integration.py")
    ds = loader.get_samples(DataStateEnum.training, list(group))

    b = len(group)
    # get_samples returns per-sample lists (never stacked) — batching to the model's B is the
    # engine's job, so grouping stays purely an I/O read-unit.
    assert isinstance(ds.inputs["image"], list) and len(ds.inputs["image"]) == b, "inputs should be a list of B"
    assert isinstance(ds.gt["classes"], list) and len(ds.gt["classes"]) == b, "gt should be a list of B"
    assert ds.inputs["image"][0].shape == (28, 28, 1), ds.inputs["image"][0].shape
    assert ds.gt["classes"][0].shape == (NUM_CLASSES,), ds.gt["classes"][0].shape
    assert list(ds.index) == list(group), f"order not preserved: {ds.index} != {group}"
    assert ds.metadata["metadata_sample_index"] == list(group)
    print(f"    get_samples returned inputs list[{b}] of {ds.inputs['image'][0].shape}, "
          f"index order preserved, metadata scattered per-sample OK")
    return ds


def _check_model_inference(ds, group):
    print("\n[4] model inference (engine batches the per-sample list) ...")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    cnn = tf.keras.models.load_model(os.path.join(dir_path, "model/model.h5"))
    # The engine assembles the model batch from the per-sample list. Here we stack the whole
    # group; a B=1-only model would instead feed each row on its own — both are possible precisely
    # because the code-loader never pre-stacked.
    batch = np.stack(ds.inputs["image"])
    preds = cnn(batch).numpy()
    assert preds.shape == (len(group), NUM_CLASSES), preds.shape
    print(f"    predictions {preds.shape}, argmax per sample: {preds.argmax(-1).tolist()}")


def check_grouped():
    responses = _check_grouped_preprocess()
    group = _check_encoders_on_group(responses[0])
    ds = _check_get_samples(group)
    _check_model_inference(ds, group)
    print("\nGrouped (list-of-inputs) validation: ALL CHECKS PASSED")


if __name__ == "__main__":
    check_grouped()
