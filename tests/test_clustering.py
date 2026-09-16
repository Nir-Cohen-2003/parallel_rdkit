import numpy as np
import pytest

import parallel_rdkit as package
from parallel_rdkit import clustering, matrix_similarity


def test_clustering_exports_and_butina():
    retained = {"butina_split", "umap_split"}
    assert retained.issubset(package.__all__)
    assert set(clustering.__all__) == retained
    assert set(matrix_similarity.__all__) == retained
    for name in retained:
        assert getattr(package, name) is getattr(clustering, name)
        assert getattr(package, name) is getattr(matrix_similarity, name)

    sim_matrix = np.array(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]],
        dtype=np.float32,
    )
    # The distance cutoff 0.3 corresponds to similarity 0.7.
    for splitter in (
        package.butina_split,
        clustering.butina_split,
        matrix_similarity.butina_split,
    ):
        labels = splitter(sim_matrix, dist_threshold=0.3)
        assert isinstance(labels, list)
        assert all(isinstance(label, int) for label in labels)
        assert len(labels) == 3
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]


def test_umap_split_preserves_input_and_forwards_options(monkeypatch):
    calls = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            calls["umap"] = kwargs

        def fit_transform(self, distances):
            calls["distances"] = distances.copy()
            return np.array([[0.0, 1.0], [1.0, 0.0]])

    class FakeKMeans:
        def __init__(self, **kwargs):
            calls["kmeans"] = kwargs

        def fit_predict(self, embedding):
            calls["embedding"] = embedding
            return np.array([1, 0])

    class FakeUmapModule:
        UMAP = FakeUMAP

    monkeypatch.setattr(clustering, "umap", FakeUmapModule)
    monkeypatch.setattr(clustering, "KMeans", FakeKMeans)
    sim_matrix = np.array([[1.0, 0.7], [0.7, 1.0]], dtype=np.float32)
    original = sim_matrix.copy()

    labels = clustering.umap_split(
        sim_matrix,
        n_clusters=2,
        random_state=17,
        n_neighbors=5,
        min_dist=0.2,
    )

    assert labels == [1, 0]
    assert isinstance(labels, list)
    np.testing.assert_array_equal(sim_matrix, original)
    np.testing.assert_allclose(calls["distances"], [[0.0, 0.3], [0.3, 0.0]])
    assert calls["umap"] == {
        "metric": "precomputed",
        "random_state": 17,
        "n_neighbors": 5,
        "min_dist": 0.2,
    }
    assert calls["kmeans"] == {
        "n_clusters": 2,
        "random_state": 17,
        "n_init": 10,
    }


@pytest.mark.parametrize("missing", ["umap", "KMeans"])
def test_umap_split_missing_optional_dependency(monkeypatch, missing):
    monkeypatch.setattr(clustering, "umap", object() if missing == "KMeans" else None)
    monkeypatch.setattr(clustering, "KMeans", object() if missing == "umap" else None)
    with pytest.raises(ImportError, match="umap-learn and scikit-learn are required"):
        clustering.umap_split(np.eye(2, dtype=np.float32))
