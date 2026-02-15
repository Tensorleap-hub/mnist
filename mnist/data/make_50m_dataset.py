"""
Generate a large (e.g. 50M) sample dataset from MNIST.

Saves data in sharded .npz files to avoid loading everything into memory.
- With --repeat-only: same MNIST samples are read multiple times (no augmentation).
- Without: each sample is an augmented version of a random MNIST training image.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Optional: use keras for loading MNIST (same as preprocess.py)
try:
    from keras.datasets import mnist as keras_mnist
except ImportError:
    keras_mnist = None


def load_mnist_train(data_dir: str | None = None):
    """Load MNIST training images and labels."""
    # Prefer existing mnist.npz if path points to file or dir containing it
    npz_path = data_dir
    if npz_path and os.path.isdir(npz_path):
        npz_path = os.path.join(npz_path, "mnist.npz")
    if npz_path and os.path.isfile(npz_path):
        data = np.load(npz_path)
        x = data["x_train"].astype(np.float32) / 255.0
        y = data["y_train"].astype(np.int32)
        return x, y
    if keras_mnist is None:
        raise ImportError("Install keras/tensorflow to load MNIST (e.g. poetry install)")
    (x_train, y_train), _ = keras_mnist.load_data()
    return x_train.astype(np.float32) / 255.0, y_train.astype(np.int32)


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply random augmentation to a single 28x28 image (float [0,1]).
    Uses only numpy for portability and speed.
    """
    out = img.copy()
    h, w = out.shape

    # Random shift: pad and crop
    pad = 4
    padded = np.pad(out, pad, mode="edge")
    dy = rng.integers(0, 2 * pad + 1)
    dx = rng.integers(0, 2 * pad + 1)
    out = padded[dy : dy + h, dx : dx + w]

    # Random zoom: crop then stretch back (nearest-neighbor via repeat)
    if rng.random() < 0.5:
        crop = rng.integers(2, 6)  # crop each side
        c = out[crop : h - crop, crop : w - crop]
        sy, sx = h / c.shape[0], w / c.shape[1]
        out = np.repeat(np.repeat(c, int(np.ceil(sy)), axis=0), int(np.ceil(sx)), axis=1)[:h, :w]

    # Additive Gaussian noise
    out = out + rng.normal(0, 0.05, out.shape).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    return out


def generate_shard(
    x_train: np.ndarray,
    y_train: np.ndarray,
    shard_size: int,
    seed: int,
    shard_index: int,
    repeat_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one shard: repeat_only=True uses same samples (no augmentation)."""
    rng = np.random.default_rng(seed + shard_index)
    n = len(x_train)
    indices = rng.integers(0, n, size=shard_size)
    if repeat_only:
        x_shard = x_train[indices]
        y_shard = y_train[indices]
        return x_shard, y_shard
    x_list = []
    for i in indices:
        x_list.append(augment_image(x_train[i], rng))
    x_shard = np.stack(x_list, axis=0)
    y_shard = y_train[indices]
    return x_shard, y_shard


def main():
    parser = argparse.ArgumentParser(description="Build 50M-sample MNIST dataset (sharded)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write shards (default: ~/tensorleap/data/mnist_50m)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000_000,
        help="Total number of samples to generate (default: 50_000_000)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Samples per .npz file (default: 100_000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mnist-cache",
        type=str,
        default=None,
        help="Path to existing mnist.npz to avoid re-download (optional)",
    )
    parser.add_argument(
        "--repeat-only",
        action="store_true",
        help="Repeat same MNIST samples (no augmentation); faster and smaller on disk",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.expanduser("~"), "tensorleap", "data", "mnist_50m"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading MNIST training set...", file=sys.stderr)
    data_path = args.mnist_cache
    if data_path and os.path.isdir(data_path):
        data_path = os.path.join(data_path, "mnist.npz")
    x_train, y_train = load_mnist_train(data_path)
    print(f"  Loaded {len(x_train)} training images.", file=sys.stderr)

    num_shards = (args.num_samples + args.shard_size - 1) // args.shard_size
    total_written = 0

    for s in range(num_shards):
        start = s * args.shard_size
        count = min(args.shard_size, args.num_samples - start)
        x_shard, y_shard = generate_shard(
            x_train, y_train, count, args.seed, s, repeat_only=args.repeat_only
        )
        # Save as (N, 28, 28) and (N,) for labels; consumer can expand_dims if needed
        path = os.path.join(args.output_dir, f"shard_{s:05d}.npz")
        np.savez_compressed(path, x=x_shard, y=y_shard)
        total_written += count
        print(f"  Wrote {path} ({count} samples, total {total_written})", file=sys.stderr)

    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"num_samples={total_written}\n")
        f.write(f"shard_size={args.shard_size}\n")
        f.write(f"num_shards={num_shards}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"repeat_only={args.repeat_only}\n")
    print(f"Done. Total samples: {total_written}. Manifest: {manifest_path}", file=sys.stderr)


def load_manifest(output_dir: str) -> dict:
    """Load manifest.txt from a generated 50M dataset directory."""
    path = os.path.join(output_dir, "manifest.txt")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k] = int(v) if v.isdigit() else v
    return out


def iter_shards(output_dir: str, shuffle_shards: bool = False, seed: int = 0):
    """
    Iterate over (x, y) arrays shard by shard from a generated 50M dataset.
    Yields (x, y) where x is (N, 28, 28) float, y is (N,) int labels.
    """
    manifest = load_manifest(output_dir)
    num_shards = manifest["num_shards"]
    order = np.arange(num_shards)
    if shuffle_shards:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for s in order:
        path = os.path.join(output_dir, f"shard_{s:05d}.npz")
        if not os.path.isfile(path):
            continue
        data = np.load(path)
        yield data["x"], data["y"]


def get_50m_dataset_info(output_dir: str) -> dict:
    """Return manifest info (num_samples, shard_size, num_shards) for the 50M dataset."""
    return load_manifest(output_dir)


if __name__ == "__main__":
    main()
