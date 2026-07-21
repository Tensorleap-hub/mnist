"""Parquet-backed MNIST preprocessing (single-sample baseline).

Sample ids are strings "{file_idx}_{local_row}", unique within a split. The
PreprocessResponse.data for a split carries the ordered list of Parquet file paths,
so an encoder can resolve a sample_id -> (file, row) and read that one sample.

Parquet files are loaded through an LRU cache: consecutive reads from the same file
hit memory, while random cross-file access thrashes the cache. That thrash is the
cost this baseline exists to expose (and that the batched-input refactor removes).
"""
import logging
import os
import time
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd

from mnist.config import CONFIG
from mnist.data.make_parquet import IMAGE_SHAPE, ensure_parquet, parquet_dir, read_manifest

logger = logging.getLogger("mnist.parquet")

_CACHE_SIZE = int(CONFIG["parquet"]["file_cache_size"])
# Counts physical Parquet reads (cache misses) over the process lifetime. Under
# random cross-file sampling this climbs far above the number of files -> the reload
# thrash the batched-input refactor is meant to remove.
_load_stats = {"loads": 0, "seconds": 0.0}

# Appends every LOAD line to a file on the shared hostPath mount (persistent across
# worker-pod churn) instead of relying on capturing stdout from short-lived pods via
# `kubectl logs`, which races pod deletion and can silently drop lines.
_LOADLOG_DIR = "/Users/orram/data/_loadlog"
_LOADLOG_PATH = os.path.join(_LOADLOG_DIR, "mnist.log")


def list_part_files(split: str) -> List[str]:
    d = parquet_dir(split)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".parquet")]


def list_sample_ids(split: str) -> List[str]:
    """Flat list of every sample id in the split, in file/row order.

    Built from the row-count manifest so no Parquet file is opened here — real
    per-sample reads (and their [parquet] LOAD instrumentation) happen only in the
    encoders. The manifest's file order matches `list_part_files` (both sorted)."""
    rows_per_file = read_manifest(split)["rows"]
    sample_ids: List[str] = []
    for file_idx, num_rows in enumerate(rows_per_file):
        sample_ids.extend(f"{file_idx}_{row}" for row in range(num_rows))
    return sample_ids


@lru_cache(maxsize=_CACHE_SIZE)
def _load_part(path: str) -> pd.DataFrame:
    """Load one Parquet file into memory. Cached (LRU) across calls.

    Only cache misses reach this body, so each log line is one physical file read.
    """
    start = time.perf_counter()
    df = pd.read_parquet(path, engine="pyarrow")
    elapsed = time.perf_counter() - start
    _load_stats["loads"] += 1
    _load_stats["seconds"] += elapsed
    msg = (f"[parquet] LOAD {os.path.basename(path)} rows={len(df)} "
           f"took={elapsed * 1000:.1f}ms total_loads={_load_stats['loads']} "
           f"total_load_time={_load_stats['seconds']:.2f}s cache_max={_CACHE_SIZE}")
    logger.info(msg)
    print(msg, flush=True)
    os.makedirs(_LOADLOG_DIR, exist_ok=True)
    with open(_LOADLOG_PATH, "a") as f:
        f.write(msg + "\n")
    return df


def _resolve(sample_id: str, preprocess) -> Tuple[pd.DataFrame, int]:
    file_idx_str, row_str = sample_id.split("_")
    path = preprocess.data["file_paths"][int(file_idx_str)]
    return _load_part(path), int(row_str)


def read_sample_image(sample_id: str, preprocess) -> np.ndarray:
    """(28, 28, 1) float32 normalized to [0, 1] — matches the original encoder output."""
    df, row = _resolve(sample_id, preprocess)
    raw = np.frombuffer(df.iloc[row]["image"], dtype=np.uint8).reshape(IMAGE_SHAPE)
    image = raw.astype("float32") / 255.0
    return np.expand_dims(image, axis=-1)


def read_sample_label(sample_id: str, preprocess) -> int:
    df, row = _resolve(sample_id, preprocess)
    return int(df.iloc[row]["label"])


def split_into_groups(sample_ids: List[str], num_groups: int) -> List[List[str]]:
    """Cut a flat sample-id list into `num_groups` disjoint, near-equal groups.

    Groups are contiguous slices of the flat list and never overlap. When the split
    doesn't divide evenly, the first `n % num_groups` groups get one extra sample.
    Used by the list-of-inputs ("grouped") path so each group is fetched/encoded as a
    single (B, *) batch instead of one sample at a time.
    """
    n = len(sample_ids)
    if num_groups < 1:
        raise ValueError("num_groups must be >= 1")
    num_groups = min(num_groups, n)  # never produce empty groups
    base, remainder = divmod(n, num_groups)
    groups: List[List[str]] = []
    start = 0
    for i in range(num_groups):
        size = base + (1 if i < remainder else 0)
        groups.append(sample_ids[start:start + size])
        start += size
    return groups


def build_split_data(split: str) -> dict:
    """PreprocessResponse.data payload for a split."""
    return {"split": split, "file_paths": list_part_files(split)}


def compute_class_centroids() -> dict:
    """Average normalized image per class, shape (28, 28, 1), keyed by label string.

    Computed from the raw source once, matching the original centroid semantics.
    """
    from mnist.data.make_parquet import _load_source
    from mnist.utils import calc_classes_centroid
    from keras.utils import to_categorical

    (x_train, y_train), _ = _load_source()
    train_x = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
    train_y = to_categorical(y_train)
    return calc_classes_centroid(train_x, train_y)


def preprocess_func(local_file_path=None):
    """Ensure the Parquet files exist and return per-split (sample_ids, data)."""
    ensure_parquet()
    return {
        split: (list_sample_ids(split), build_split_data(split))
        for split in ("train", "val")
    }
