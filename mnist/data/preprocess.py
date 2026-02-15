from typing import Union, Iterable, Tuple
import numpy as np
from numpy import ndarray
from keras.datasets import mnist
from keras.utils import to_categorical
import os

from mnist.config import CONFIG

try:
    from mnist.data.make_50m_dataset import load_manifest
except ImportError:
    load_manifest = None


def get_full_path(path):
    if path is None:
        path = CONFIG['local_file_path']
    if os.path.isabs(path):
        return path.rstrip(os.sep)
    elif 'IS_CLOUD' in os.environ and os.path.exists("/nfs"):  # SaaS env
        return os.path.join("/nfs", path)
    elif 'GENERIC_HOST_PATH' in os.environ:  # OnPrem
        return os.path.join(os.environ['GENERIC_HOST_PATH'], path)
    else:
        # e.g. "mnist_50m" -> ~/tensorleap/data/mnist_50m
        # e.g. "tensorleap/data/mnist_50m" -> ~/tensorleap/data/mnist_50m
        path = path.rstrip(os.sep)
        if path.startswith('tensorleap/data') or path.startswith('tensorleap' + os.sep + 'data'):
            return os.path.join(os.path.expanduser('~'), path)
        return os.path.join(os.path.expanduser('~'), 'tensorleap', 'data', path)


def _is_sharded_dataset(dir_path: str) -> bool:
    """True if dir_path contains manifest.txt and at least one shard_*.npz."""
    if not dir_path or not os.path.isdir(dir_path):
        return False
    manifest = os.path.join(dir_path, 'manifest.txt')
    if not os.path.isfile(manifest):
        return False
    # Check for at least one shard
    for name in os.listdir(dir_path):
        if name.startswith('shard_') and name.endswith('.npz'):
            return True
    return False


class _ShardLoader:
    """Loads samples by index from sharded .npz files; caches one shard at a time."""

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.manifest = load_manifest(dir_path) if load_manifest else {}
        self._num_samples = int(self.manifest.get('num_samples', 0))
        self._shard_size = int(self.manifest.get('shard_size', 0))
        self._num_shards = int(self.manifest.get('num_shards', 0))
        self._cached_shard_idx = -1
        self._cached_x = None
        self._cached_y = None

    def __len__(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> Tuple[ndarray, ndarray]:
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(idx)
        shard_idx = idx // self._shard_size
        local_idx = idx % self._shard_size
        if shard_idx != self._cached_shard_idx:
            path = os.path.join(self.dir_path, f'shard_{shard_idx:05d}.npz')
            data = np.load(path)
            self._cached_x = data['x']
            self._cached_y = data['y']
            self._cached_shard_idx = shard_idx
        return self._cached_x[local_idx], self._cached_y[local_idx]


class _LazyShardedImages:
    """Indexable view of images from sharded dataset; returns (28, 28, 1) float32 in [0, 1]."""

    def __init__(self, loader: _ShardLoader):
        self._loader = loader

    def __len__(self) -> int:
        return len(self._loader)

    def __getitem__(self, idx: int) -> ndarray:
        x, _ = self._loader.get(idx)
        x = np.expand_dims(x, axis=-1).astype(np.float32)
        return x


class _LazyShardedLabels:
    """Indexable view of labels from sharded dataset; returns one-hot float32."""

    def __init__(self, loader: _ShardLoader):
        self._loader = loader

    def __len__(self) -> int:
        return len(self._loader)

    def __getitem__(self, idx: int) -> ndarray:
        _, y = self._loader.get(idx)
        return to_categorical(y, num_classes=10).astype(np.float32)


def _load_validation_from_mnist(local_file_path: str) -> Tuple[ndarray, ndarray]:
    """Load validation (test) set from mnist.npz in same dir or keras default."""
    data_file = os.path.join(local_file_path, 'mnist.npz')
    if os.path.isfile(data_file):
        data = np.load(data_file)
        val_X = data['x_test'].astype(np.float32) / 255.0
        val_Y = data['y_test']
    else:
        (_, _), (val_X, val_Y) = mnist.load_data()
        val_X = val_X.astype(np.float32) / 255.0
    val_X = np.expand_dims(val_X, axis=-1)
    val_Y = to_categorical(val_Y)
    return val_X, val_Y


def _centroid_from_first_shard(dir_path: str) -> dict:
    """Compute class centroids from the first shard (for metadata)."""
    from mnist.utils import calc_classes_centroid
    path = os.path.join(dir_path, 'shard_00000.npz')
    data = np.load(path)
    x = np.expand_dims(data['x'], axis=-1).astype(np.float32)
    y = to_categorical(data['y'])
    return calc_classes_centroid(x, y)


def preprocess_func(local_file_path) -> Union[ndarray, Iterable, int, float, tuple, dict]:
    # Resolve path
    if local_file_path is None:
        local_file_path = CONFIG['local_file_path']
    local_file_path = get_full_path(local_file_path)
    if not os.path.exists(local_file_path):
        os.makedirs(local_file_path)

    # Sharded dataset (multiple shard_*.npz + manifest.txt)
    if load_manifest and _is_sharded_dataset(local_file_path):
        loader = _ShardLoader(local_file_path)
        train_X = _LazyShardedImages(loader)
        train_Y = _LazyShardedLabels(loader)
        val_X, val_Y = _load_validation_from_mnist(local_file_path)
        classes_avg_images = _centroid_from_first_shard(local_file_path)
        return train_X, val_X, train_Y, val_Y, classes_avg_images

    # Single-file dataset (mnist.npz)
    data_file = os.path.join(local_file_path, 'mnist.npz')
    if not os.path.exists(data_file):
        (train_X, train_Y), (val_X, val_Y) = mnist.load_data(data_file)
    else:
        data = np.load(data_file)
        train_X = data['x_train']
        val_X = data['x_test']
        train_Y = data['y_train']
        val_Y = data['y_test']

    train_X = np.expand_dims(train_X, axis=-1).astype(np.float32)
    train_X = train_X / 255.0
    train_Y = to_categorical(train_Y).astype(np.float32)

    val_X = np.expand_dims(val_X, axis=-1).astype(np.float32)
    val_X = val_X / 255.0
    val_Y = to_categorical(val_Y).astype(np.float32)

    return train_X, val_X, train_Y, val_Y