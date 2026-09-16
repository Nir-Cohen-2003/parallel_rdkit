"""Generic clustering helpers for similarity matrices."""

from typing import List

import numpy as np
from rdkit.ML.Cluster import Butina

try:
    import umap
    from sklearn.cluster import KMeans
except ImportError:
    umap = None
    KMeans = None

__all__ = ["butina_split", "umap_split"]


def butina_split(
    sim_matrix: np.ndarray,
    dist_threshold: float = 0.3,
) -> List[int]:
    """Perform Butina clustering on a similarity matrix."""
    n_mols = sim_matrix.shape[0]
    # Butina expects a distance matrix (lower triangle flattened)
    dists = []
    for i in range(1, n_mols):
        for j in range(i):
            dists.append(1.0 - float(sim_matrix[i, j]))

    clusters = Butina.ClusterData(dists, n_mols, dist_threshold, isDistData=True)

    labels = np.zeros(n_mols, dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for mol_idx in cluster:
            labels[mol_idx] = cluster_id

    return labels.tolist()


def umap_split(
    sim_matrix: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
    **umap_kwargs
) -> List[int]:
    """Perform UMAP reduction followed by KMeans clustering."""
    if umap is None or KMeans is None:
        raise ImportError("umap-learn and scikit-learn are required for umap_split")

    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    reducer = umap.UMAP(
        metric='precomputed',
        random_state=random_state,
        **umap_kwargs
    )
    embedding = reducer.fit_transform(dist_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embedding)

    return labels.tolist()
