from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, norms, out=x, where=norms > 0)
    return x


def _make_dataset(
    *,
    train_size: int,
    query_size: int,
    dim: int,
    latent_dim: int,
    n_centers: int,
    cluster_noise: float,
    duplicate_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_size < 1000:
        raise ValueError("train_size must be >= 1000")
    if query_size < 100:
        raise ValueError("query_size must be >= 100")
    if dim <= 8:
        raise ValueError("dim must be > 8")
    if latent_dim < 8 or latent_dim > dim:
        raise ValueError("latent_dim must be in [8, dim]")
    if n_centers < 8:
        raise ValueError("n_centers must be >= 8")
    if cluster_noise <= 0.0:
        raise ValueError("cluster_noise must be > 0")
    if not (0.0 <= duplicate_fraction < 1.0):
        raise ValueError("duplicate_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)

    # Many nearby clusters in latent space create overlapping neighborhoods.
    centers = rng.normal(size=(n_centers, latent_dim)).astype(np.float32)
    centers = _normalize_rows(centers)

    train_assign = rng.integers(0, n_centers, size=train_size)
    query_assign = rng.integers(0, n_centers, size=query_size)

    train_latent = centers[train_assign] + rng.normal(
        scale=cluster_noise, size=(train_size, latent_dim)
    ).astype(np.float32)
    query_latent = centers[query_assign] + rng.normal(
        scale=cluster_noise * 1.1, size=(query_size, latent_dim)
    ).astype(np.float32)

    # Random projection to higher-dimensional observed space.
    projection = rng.normal(size=(latent_dim, dim)).astype(np.float32)
    projection /= np.sqrt(float(latent_dim))
    train = train_latent @ projection
    queries = query_latent @ projection

    # Anisotropic scaling makes distance landscape less uniform.
    scales = np.exp(rng.normal(loc=0.0, scale=0.6, size=(dim,))).astype(np.float32)
    train *= scales
    queries *= scales

    # Add near-duplicates to increase local crowding.
    dup_count = int(train_size * duplicate_fraction)
    if dup_count > 0:
        src = rng.integers(0, train_size, size=dup_count)
        dst = rng.choice(train_size, size=dup_count, replace=False)
        train[dst] = train[src] + rng.normal(scale=0.005, size=(dup_count, dim)).astype(np.float32)

    train = _normalize_rows(train.astype(np.float32))
    queries = _normalize_rows(queries.astype(np.float32))
    return train, queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ambiguous angular dataset with crowded neighborhoods."
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--train-size", type=int, default=120_000)
    parser.add_argument("--query-size", type=int, default=8_000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--n-centers", type=int, default=512)
    parser.add_argument("--cluster-noise", type=float, default=0.22)
    parser.add_argument("--duplicate-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    train, queries = _make_dataset(
        train_size=args.train_size,
        query_size=args.query_size,
        dim=args.dim,
        latent_dim=args.latent_dim,
        n_centers=args.n_centers,
        cluster_noise=args.cluster_noise,
        duplicate_fraction=args.duplicate_fraction,
        seed=args.seed,
    )

    np.savez(
        output,
        train=train,
        queries=queries,
        distance="angular",
    )

    print(f"written: {output.resolve()}")
    print(f"train={train.shape}, queries={queries.shape}, metric=angular")


if __name__ == "__main__":
    main()
