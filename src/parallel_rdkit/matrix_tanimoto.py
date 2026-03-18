import gc
import logging
import math
import threading
import traceback
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.ML.Cluster import Butina

from .mol import sanitize_smiles
from .fingerprint import FingerprintParams

try:
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
    NVMOLKIT_AVAILABLE = True
except ImportError:
    NVMOLKIT_AVAILABLE = False
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
    """
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


def _generate_fingerprints_gpu(
    smiles: List[str],
    fp_params: FingerprintParams,
    log_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """
    Generate fingerprints using nvmolkit GPU acceleration.
    
    Returns:
        tuple: (fps_gpu, valid_indices, valid_smiles)
            - fps_gpu: GPU tensor of fingerprints
            - valid_indices: list of original indices that are valid
            - valid_smiles: list of valid SMILES strings
    """
    if not NVMOLKIT_AVAILABLE:
        raise ImportError("nvmolkit is required for GPU fingerprint generation")
    
    # Sanitize SMILES
    if log_path:
        _log_message_to_file(f"Sanitizing {len(smiles)} SMILES...", log_path)
    
    batch_size_san = 1 + len(smiles) // 6
    sanitized = sanitize_smiles(smiles, batch_size=batch_size_san)
    
    # Map back to valid molecules only
    valid_indices = [i for i, s in enumerate(sanitized) if s]
    valid_smiles = [sanitized[i] for i in valid_indices]
    
    if not valid_smiles:
        if log_path:
            _log_message_to_file("No valid molecules found in input.", log_path, level=logging.ERROR)
        raise ValueError("No valid molecules found in input")

    if len(valid_smiles) != len(smiles):
        if log_path:
            _log_message_to_file(
                f"Warning: {len(smiles) - len(valid_smiles)} invalid molecules skipped. "
                f"Processing {len(valid_smiles)} valid molecules.",
                log_path,
                level=logging.WARNING
            )

    # Convert to RDKit Mols
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    
    # Generate fingerprints on GPU
    if log_path:
        _log_message_to_file("Generating fingerprints on GPU...", log_path)
    
    fpgen = MorganFingerprintGenerator(radius=fp_params.radius, fpSize=fp_params.fpSize)
    fps_gpu = fpgen.GetFingerprints(mols)
    
    return fps_gpu, valid_indices, valid_smiles


def _write_similarity_chunk_to_parquet(
    sim_matrix_chunk: np.ndarray,
    row_start: int,
    col_indices: np.ndarray,
    parquet_writer: pq.ParquetWriter,
    threshold: Optional[float] = None,
) -> int:
    """
    Write a chunk of similarity matrix to parquet, storing only lower triangular (i <= j).
    Applies threshold filtering on GPU before returning data.
    
    Args:
        sim_matrix_chunk: Similarity matrix chunk (n_rows x n_cols)
        row_start: Starting row index for this chunk
        col_indices: Column indices corresponding to sim_matrix_chunk columns
        parquet_writer: PyArrow ParquetWriter instance
        threshold: Optional minimum similarity threshold (only store >= threshold)
        
    Returns:
        Number of rows written
    """
    n_rows, n_cols = sim_matrix_chunk.shape
    
    # Build coordinate arrays
    row_indices = np.arange(row_start, row_start + n_rows, dtype=np.uint32)
    
    # For each row, get columns where col_idx >= row_idx (lower triangular)
    mol1_list = []
    mol2_list = []
    tanimoto_list = []
    
    for i, row_idx in enumerate(row_indices):
        # Get valid column indices for this row (lower triangular only: j >= i)
        valid_mask = col_indices >= row_idx
        if not np.any(valid_mask):
            continue
            
        row_vals = sim_matrix_chunk[i, valid_mask]
        col_vals = col_indices[valid_mask]
        
        # Apply threshold filtering
        if threshold is not None:
            thresh_mask = row_vals >= threshold
            row_vals = row_vals[thresh_mask]
            col_vals = col_vals[thresh_mask]
        
        if len(row_vals) > 0:
            mol1_list.extend([row_idx] * len(row_vals))
            mol2_list.extend(col_vals.tolist())
            tanimoto_list.extend(row_vals.tolist())
    
    if not mol1_list:
        return 0
    
    # Create PyArrow table
    table = pa.table({
        "mol1_idx": pa.array(mol1_list, type=pa.uint32()),
        "mol2_idx": pa.array(mol2_list, type=pa.uint32()),
        "tanimoto": pa.array(tanimoto_list, type=pa.float32()),
    })
    
    parquet_writer.write_table(table)
    
    return len(mol1_list)


def _compute_tanimoto_matrix_chunked_parquet(
    fps_gpu,
    indices: np.ndarray,
    parquet_path: Union[str, Path],
    log_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 5000,
    threshold: Optional[float] = None,
) -> int:
    """
    Compute Tanimoto matrix in chunks and write to parquet file.
    Stores only lower triangular (i <= j) with optional threshold filtering.
    
    Args:
        fps_gpu: GPU fingerprint tensor
        indices: Original indices of valid molecules
        parquet_path: Path to output parquet file
        log_path: Path to log file
        chunk_size: Number of rows to process per chunk
        threshold: Minimum similarity threshold
        
    Returns:
        Total number of similarity pairs written
    """
    n_mols = len(indices)
    total_written = 0
    
    if log_path:
        _log_message_to_file(
            f"Computing Tanimoto matrix in {math.ceil(n_mols / chunk_size)} chunks "
            f"(chunk_size={chunk_size}, n_mols={n_mols})...", 
            log_path
        )
        if threshold:
            _log_message_to_file(f"Applying similarity threshold: {threshold}", log_path)
    
    # Define parquet schema
    schema = pa.schema([
        ("mol1_idx", pa.uint32()),
        ("mol2_idx", pa.uint32()),
        ("tanimoto", pa.float32()),
    ])
    
    # Open parquet writer with snappy compression
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pq.ParquetWriter(
        str(parquet_path),
        schema,
        compression="snappy",
        use_dictionary=["mol1_idx", "mol2_idx"],  # Enable dictionary encoding for indices
    ) as writer:
        
        # Process row chunks
        for row_start in range(0, n_mols, chunk_size):
            row_end = min(row_start + chunk_size, n_mols)
            chunk_start_time = perf_counter()
            
            # Get fingerprint chunk for rows
            # Note: nvmolkit expects specific format, we pass appropriate slice
            row_fps_chunk = fps_gpu[row_start:row_end]
            
            # Compute similarity: chunk x all molecules
            # crossTanimotoSimilarityMemoryConstrained handles GPU chunking internally
            sim_chunk = crossTanimotoSimilarityMemoryConstrained(row_fps_chunk, fps_gpu)
            sim_chunk = np.asarray(sim_chunk, dtype=np.float32)
            
            # Write to parquet (lower triangular with threshold filtering)
            rows_written = _write_similarity_chunk_to_parquet(
                sim_chunk,
                indices[row_start],
                indices,  # column indices are the full set
                writer,
                threshold=threshold,
            )
            total_written += rows_written
            
            chunk_time = perf_counter() - chunk_start_time
            if log_path:
                _log_message_to_file(
                    f"Chunk {row_start//chunk_size + 1}/{(n_mols-1)//chunk_size + 1}: "
                    f"rows {row_start}-{row_end-1}, "
                    f"wrote {rows_written} pairs, "
                    f"took {chunk_time:.2f}s",
                    log_path
                )
            
            # Cleanup
            del sim_chunk
            gc.collect()
    
    if log_path:
        _log_message_to_file(
            f"Parquet writing complete. Total pairs written: {total_written}",
            log_path
        )
    
    return total_written


def calculate_tanimoto_matrix(
    smiles: List[str],
    indices: Optional[np.ndarray] = None,
    fp_params: Optional[FingerprintParams] = None,
    output_mode: Literal["numpy", "parquet"] = "numpy",
    parquet_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    chunk_size: int = 5000,
    log_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]:
    """
    Calculate Tanimoto similarity matrix for a list of SMILES strings using GPU.
    
    Two output modes:
    - "numpy": Returns dense matrix in memory (backward compatible)
    - "parquet": Writes lower-triangular similarities to parquet file (memory efficient)
    
    Args:
        smiles: List of SMILES strings.
        indices: Original indices for molecules (caller-provided). If None, uses range(len(smiles)).
        fp_params: Fingerprint parameters (FingerprintParams dataclass). Defaults to Morgan r=2, size=2048.
        output_mode: "numpy" or "parquet".
        parquet_path: Required for parquet mode. Path to output parquet file.
        threshold: Minimum similarity threshold for parquet mode (only store similarities >= threshold).
        chunk_size: Number of rows per chunk for parquet mode computation.
        log_path: Path to log progress. If None and parquet_path is provided, uses parquet_path with .log suffix.
        
    Returns:
        For "numpy" mode: NumPy array of shape (N, N) containing similarity scores.
        For "parquet" mode: None (results written to file).
        
    Raises:
        ValueError: If invalid mode or missing required parameters.
        ImportError: If nvmolkit is not available for GPU acceleration.
    """
    if not NVMOLKIT_AVAILABLE:
        raise ImportError("nvmolkit is required for GPU Tanimoto calculation")
    
    if output_mode not in ("numpy", "parquet"):
        raise ValueError(f"output_mode must be 'numpy' or 'parquet', got '{output_mode}'")
    
    if output_mode == "parquet" and parquet_path is None:
        raise ValueError("parquet_path is required when output_mode='parquet'")
    
    # Auto-generate log path if not provided
    if log_path is None and parquet_path is not None:
        parquet_path_obj = Path(parquet_path)
        log_path = parquet_path_obj.with_suffix('.log')
    
    # Default fingerprint parameters
    if fp_params is None:
        fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)
    
    # Default indices
    if indices is None:
        indices = np.arange(len(smiles), dtype=np.uint32)
    else:
        indices = np.asarray(indices, dtype=np.uint32)
    
    start = perf_counter()
    
    if log_path:
        _log_message_to_file(
            f"Starting Tanimoto matrix calculation for {len(smiles)} molecules. "
            f"Output mode: {output_mode}",
            log_path,
            overwrite=True
        )
    
    try:
        # Generate fingerprints on GPU
        fps_gpu, valid_indices, valid_smiles = _generate_fingerprints_gpu(
            smiles, fp_params, log_path
        )
        
        # Filter indices to match valid molecules
        valid_indices_arr = np.array(valid_indices, dtype=np.uint32)
        indices = indices[valid_indices_arr]
        
        if output_mode == "parquet":
            # Compute in chunks and write to parquet
            _compute_tanimoto_matrix_chunked_parquet(
                fps_gpu,
                indices,
                parquet_path,
                log_path=log_path,
                chunk_size=chunk_size,
                threshold=threshold,
            )
            
            end = perf_counter()
            if log_path:
                _log_message_to_file(
                    f"Parquet calculation completed in {end - start:.2f}s. "
                    f"Output: {parquet_path}",
                    log_path
                )
            
            return None
            
        else:  # numpy mode
            if log_path:
                _log_message_to_file("Computing full Tanimoto similarity matrix on GPU...", log_path)
            
            # Compute full matrix (may OOM for large datasets)
            sim_matrix = crossTanimotoSimilarityMemoryConstrained(fps_gpu, fps_gpu)
            sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
            
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
    index_column: str = "index",
    fp_params: Optional[FingerprintParams] = None,
    output_mode: Literal["numpy", "parquet"] = "parquet",
    output_parquet_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    chunk_size: int = 5000,
    log_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]:
    """
    Calculate Tanimoto matrix by reading SMILES from a parquet file using polars streaming.
    
    Args:
        parquet_path: Path to input parquet file containing SMILES.
        smiles_column: Name of column containing SMILES strings.
        index_column: Name of column containing molecule indices.
        fp_params: Fingerprint parameters.
        output_mode: "numpy" or "parquet".
        output_parquet_path: Path for output parquet (required for parquet mode).
        threshold: Minimum similarity threshold for parquet mode.
        chunk_size: Number of rows per chunk.
        log_path: Path to log file.
        
    Returns:
        For numpy mode: similarity matrix array.
        For parquet mode: None.
    """
    if log_path is None and output_parquet_path is not None:
        log_path = Path(output_parquet_path).with_suffix('.log')
    
    if log_path:
        _log_message_to_file(
            f"Reading SMILES from {parquet_path} using polars.",
            log_path,
            overwrite=True
        )
    
    # Read SMILES and indices from parquet
    df = pl.scan_parquet(str(parquet_path)).select([
        pl.col(smiles_column),
        pl.col(index_column)
    ]).collect()
    
    smiles_list = df.get_column(smiles_column).to_list()
    indices = df.get_column(index_column).to_numpy().astype(np.uint32)
    
    return calculate_tanimoto_matrix(
        smiles=smiles_list,
        indices=indices,
        fp_params=fp_params,
        output_mode=output_mode,
        parquet_path=output_parquet_path,
        threshold=threshold,
        chunk_size=chunk_size,
        log_path=log_path,
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
