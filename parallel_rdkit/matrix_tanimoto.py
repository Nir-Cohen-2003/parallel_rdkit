import logging
import math
import tempfile
import threading
import traceback
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.ML.Cluster import Butina

from .mol import sanitize_smiles

try:
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
except ImportError:
    # These are expected to be available in the environment where this runs
    logger = logging.getLogger(__name__)
    logger.warning("nvmolkit not found. GPU acceleration will not be available.")

# Optional dependencies for UMAP split
try:
    import umap
    from sklearn.cluster import KMeans
except ImportError:
    umap = None
    KMeans = None

# Tracks which log files have been truncated for this process so we only truncate once.
_initialized_log_paths: set[str] = set()
_initialized_log_paths_lock = threading.Lock()

# Module logger
logger = logging.getLogger(__name__)


def _log_message_to_file(
    message: str,
    log_path: Union[str, Path],
    level: int = logging.INFO,
    overwrite: bool = False,
) -> None:
    """
    Log a single message to a file by attaching a temporary FileHandler to the module logger.
    Reused from parallel_rdkit.tanimoto.
    """
    logger = logging.getLogger(__name__)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    canonical = str(Path(log_path).resolve(strict=False))
    if overwrite:
        with _initialized_log_paths_lock:
            if canonical in _initialized_log_paths:
                do_truncate = False
            else:
                do_truncate = True
                _initialized_log_paths.add(canonical)
    else:
        do_truncate = False

    mode = "w" if do_truncate else "a"
    handler = logging.FileHandler(str(log_path), mode=mode)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    prev_level = logger.level
    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    try:
        logger.log(level, message)
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(prev_level)


def calculate_tanimoto_matrix(
    smiles: List[str],
    fp_radius: int = 2,
    fp_size: int = 2048,
    save_path: Optional[Union[str, Path]] = None,
    log_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Calculate a dense Tanimoto similarity matrix for a list of SMILES strings using GPU.
    
    Args:
        smiles: List of SMILES strings.
        fp_radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint bit size.
        save_path: Path to save the resulting matrix as a .npy file.
        log_path: Path to log progress.
        
    Returns:
        A NumPy array of shape (N, N) containing similarity scores.
    """
    if log_path:
        _log_message_to_file(
            f"Starting Tanimoto matrix calculation for {len(smiles)} molecules.",
            log_path,
            overwrite=True
        )

    start = perf_counter()
    
    # 1. Sanitize and Canonicalize
    if log_path:
        _log_message_to_file("Sanitizing SMILES...", log_path)
    
    # Using the same batching logic as tanimoto.py
    batch_size_san = 1 + len(smiles) // 6
    sanitized = sanitize_smiles(smiles, batch_size=batch_size_san)
    
    # Map back to valid molecules only
    valid_indices = [i for i, s in enumerate(sanitized) if s]
    valid_smiles = [sanitized[i] for i in valid_indices]
    
    if not valid_smiles:
        if log_path:
            _log_message_to_file("No valid molecules found in input.", log_path, level=logging.ERROR)
        return np.array([[]], dtype=np.float32)

    if len(valid_smiles) != len(smiles):
        if log_path:
            _log_message_to_file(
                f"Warning: {len(smiles) - len(valid_smiles)} invalid molecules skipped. "
                f"Resulting matrix will be {len(valid_smiles)}x{len(valid_smiles)}.",
                log_path,
                level=logging.WARNING
            )

    # Convert to RDKit Mols
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles]

    try:
        # 2. Fingerprint generation (GPU)
        if log_path:
            _log_message_to_file("Generating fingerprints on GPU...", log_path)
        
        fpgen = MorganFingerprintGenerator(radius=fp_radius, fpSize=fp_size)
        fps = fpgen.GetFingerprints(mols)
        
        # 3. Similarity matrix computation (GPU)
        if log_path:
            _log_message_to_file("Computing Tanimoto similarity matrix on GPU...", log_path)
            
        # Using memory constrained routine for large molecule sets
        # This computes the self-similarity matrix
        sim_matrix = crossTanimotoSimilarityMemoryConstrained(fps, fps)
        
        # Ensure it's a numpy array
        sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), sim_matrix)
            if log_path:
                _log_message_to_file(f"Matrix saved to {save_path}", log_path)

        end = perf_counter()
        if log_path:
            _log_message_to_file(
                f"Matrix calculation completed in {end - start:.2f}s. "
                f"Shape: {sim_matrix.shape}",
                log_path
            )
            
        return sim_matrix

    except Exception:
        err = traceback.format_exc()
        if log_path:
            _log_message_to_file(f"Error in matrix calculation:\n{err}", log_path, level=logging.ERROR)
        raise


def calculate_tanimoto_matrix_streaming(
    parquet_path: Union[str, Path],
    smiles_column: str = "smiles",
    save_path: Optional[Union[str, Path]] = None,
    log_path: Optional[Union[str, Path]] = None,
    fp_radius: int = 2,
    fp_size: int = 2048,
) -> np.ndarray:
    """
    Calculate Tanimoto matrix by reading SMILES from a parquet file using polars streaming.
    """
    if log_path:
        _log_message_to_file(f"Reading SMILES from {parquet_path} using polars streaming.", log_path)
    
    # Read SMILES efficiently
    df = pl.scan_parquet(str(parquet_path)).select(pl.col(smiles_column)).collect()
    smiles_list = df.get_column(smiles_column).to_list()
    
    return calculate_tanimoto_matrix(
        smiles_list,
        fp_radius=fp_radius,
        fp_size=fp_size,
        save_path=save_path,
        log_path=log_path
    )


def butina_split(
    sim_matrix: np.ndarray,
    dist_threshold: float = 0.3,
) -> List[int]:
    """
    Perform Butina clustering on the similarity matrix.
    
    Args:
        sim_matrix: Dense similarity matrix (N x N).
        dist_threshold: Distance threshold (1 - similarity) for clustering.
        
    Returns:
        List of cluster IDs for each molecule in the matrix.
    """
    n_mols = sim_matrix.shape[0]
    # Butina expects a distance matrix (lower triangle flattened)
    dists = []
    for i in range(1, n_mols):
        for j in range(i):
            dists.append(1.0 - float(sim_matrix[i, j]))
            
    clusters = Butina.ClusterData(dists, n_mols, dist_threshold, isDistData=True)
    
    # Map back to cluster labels
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
    """
    Perform UMAP reduction followed by KMeans clustering for splitting.
    
    Args:
        sim_matrix: Similarity matrix.
        n_clusters: Number of clusters for KMeans.
        random_state: Seed for reproducibility.
        **umap_kwargs: Additional arguments for UMAP (e.g., n_neighbors, min_dist).
        
    Returns:
        List of cluster labels.
    """
    if umap is None or KMeans is None:
        raise ImportError("umap-learn and scikit-learn are required for umap_split")
    
    # UMAP works better with distance
    dist_matrix = 1.0 - sim_matrix
    # Ensure diagonal is exactly 0 for distance (some float precision issues might exist)
    np.fill_diagonal(dist_matrix, 0.0)
    
    # UMAP using precomputed distance matrix
    # Default metric for UMAP when passing a distance matrix is 'precomputed'
    reducer = umap.UMAP(
        metric='precomputed', 
        random_state=random_state, 
        **umap_kwargs
    )
    embedding = reducer.fit_transform(dist_matrix)
    
    # KMeans on the resulting embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embedding)
    
    return labels.tolist()
