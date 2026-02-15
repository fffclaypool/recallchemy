import numpy as np

from recallchemy.dataset import load_dataset


def test_load_dataset_keeps_query_ground_truth_alignment(tmp_path):
    train = np.zeros((20, 4), dtype=np.float32)
    queries = np.array([[float(i), 1.0, 2.0, 3.0] for i in range(10)], dtype=np.float32)
    neighbors = np.array([[i, (i + 1) % 20, (i + 2) % 20] for i in range(10)], dtype=np.int64)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, train=train, queries=queries, neighbors=neighbors, distance="euclidean")

    bundle = load_dataset(dataset_path, metric=None, max_queries=4, seed=123)
    assert bundle.ground_truth is not None
    assert bundle.queries.shape[0] == 4
    assert bundle.ground_truth.shape[0] == 4

    for query_vec, gt in zip(bundle.queries, bundle.ground_truth):
        row_id = int(query_vec[0])
        assert gt[0] == row_id

