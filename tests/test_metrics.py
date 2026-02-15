import numpy as np

from recallchemy.metrics import compute_ground_truth, recall_at_k


def test_compute_ground_truth_euclidean_shape():
    train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    queries = np.array([[0.2, 0.1], [1.8, 1.9]], dtype=np.float32)
    gt = compute_ground_truth(train=train, queries=queries, k=2, metric="euclidean", batch_size=1)
    assert gt.shape == (2, 2)
    assert gt[0, 0] == 0
    assert gt[1, 0] == 2


def test_recall_at_k():
    predictions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    ground_truth = np.array([[1, 9, 8], [7, 5, 6]], dtype=np.int64)
    # Hits: first row -> {1} (1 hit), second row -> {5,6} (2 hits), total = 3/6.
    recall = recall_at_k(predictions, ground_truth, k=3)
    assert abs(recall - 0.5) < 1e-9

