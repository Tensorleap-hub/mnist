"""Generate Parquet-backed MNIST storage for the baseline example.

The dataset is split into many small Parquet files, each holding
`CONFIG['parquet']['samples_per_file']` samples. Each such file is one "group":
the unit that the batched-input refactor will later hand to the encoders together.

For this (single-sample) baseline the files are still read one sample at a time,
which is exactly the inefficient access pattern the refactor is meant to remove.

Each Parquet row:
    sample_id: str   "{file_idx}_{local_row}"   (unique within a split)
    image:     bytes  raw uint8, 28*28 = 784 bytes
    label:     int    0..9

Run manually with:  poetry run python -m mnist.data.make_parquet
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd

from mnist.config import CONFIG

IMAGE_SHAPE = (28, 28)
SPLITS = ("train", "val")


def data_root() -> str:
    """Root directory that holds the mnist data, matching how the engine mounts it."""
    name = CONFIG["local_file_path"]  # "mnist"
    if os.path.isabs(name):
        return name
    if "IS_CLOUD" in os.environ and os.path.exists("/nfs"):  # SaaS
        return os.path.join("/nfs", name)
    if "GENERIC_HOST_PATH" in os.environ:  # OnPrem
        return os.path.join(os.environ["GENERIC_HOST_PATH"], name)
    return os.path.join(os.path.expanduser("~"), "tensorleap", "data", name)


def parquet_dir(split: str) -> str:
    return os.path.join(data_root(), "parquet", split)


def _load_source() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load raw uint8 MNIST from the in-repo npz, falling back to a keras download."""
    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, "mnist.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    from keras.datasets import mnist
    return mnist.load_data()


def _write_split(split: str, images: np.ndarray, labels: np.ndarray) -> int:
    """Write one split into part_XXXX.parquet files of `samples_per_file` rows each."""
    samples_per_file = int(CONFIG["parquet"]["samples_per_file"])
    out_dir = parquet_dir(split)
    os.makedirs(out_dir, exist_ok=True)

    images = images.astype(np.uint8).reshape(len(images), -1)  # (N, 784)
    labels = labels.astype(int)

    num_files = (len(images) + samples_per_file - 1) // samples_per_file
    for file_idx in range(num_files):
        start = file_idx * samples_per_file
        end = min(start + samples_per_file, len(images))
        rows = {
            "sample_id": [f"{file_idx}_{row}" for row in range(end - start)],
            "image": [images[i].tobytes() for i in range(start, end)],
            "label": [int(labels[i]) for i in range(start, end)],
        }
        path = os.path.join(out_dir, f"part_{file_idx:04d}.parquet")
        pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    return num_files


def generate_parquet_files(force: bool = False) -> None:
    (x_train, y_train), (x_val, y_val) = _load_source()
    per_split = {"train": (x_train, y_train), "val": (x_val, y_val)}
    for split in SPLITS:
        if not force and os.path.isdir(parquet_dir(split)) and _list_parts(split):
            print(f"[make_parquet] {split}: already present, skipping")
            continue
        images, labels = per_split[split]
        n = _write_split(split, images, labels)
        print(f"[make_parquet] {split}: wrote {n} files ({len(images)} samples) -> {parquet_dir(split)}")


def _list_parts(split: str):
    d = parquet_dir(split)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".parquet"))


def ensure_parquet() -> None:
    """Generate the Parquet files on first use if they are missing."""
    if all(_list_parts(split) for split in SPLITS):
        return
    generate_parquet_files(force=False)


if __name__ == "__main__":
    generate_parquet_files(force=True)
