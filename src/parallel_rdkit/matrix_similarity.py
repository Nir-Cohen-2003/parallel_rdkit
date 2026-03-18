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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .mol import sanitize_smiles
from .fingerprint import FingerprintParams

try:
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import (
        crossTanimotoSimilarityMemoryConstrained,
        crossCosineSimilarityMemoryConstrained,
    )
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
        tuple: (fps_tensor, valid_indices, valid_smiles)
            - fps_tensor: torch.Tensor of fingerprints on GPU (from AsyncGpuResult.torch())
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
    
    # Convert AsyncGpuResult to torch.Tensor
    # This is the key fix - AsyncGpuResult is not subscriptable, but torch.Tensor is
    fps_tensor = fps_gpu.torch()
    
    return fps_tensor, valid_indices, valid_smiles


def _calculate_square_chunk_size(n_mols: int, memory_usage_fraction: float = 0.5, storage_mode: str = "single") -> int:
    """
    Calculate square chunk dimension based on available CPU memory.
    
    For square chunking, we process d x d submatrices at a time.
    The similarity data is stored as triplets in parquet:
    - Single similarity: 12 bytes per pair (mol1_idx + mol2_idx + similarity)
    - Both similarities: 16 bytes per pair (mol1_idx + mol2_idx + tanimoto + cosine)
    
    For lower triangular storage, each d x d block stores at most d*(d+1)/2 pairs.
    
    Args:
        n_mols: Total number of molecules
        memory_usage_fraction: Fraction of available memory to use (0.0-1.0)
        storage_mode: "single" for one similarity metric, "both" for tanimoto+cosine
        
    Returns:
        Square dimension d for d x d chunks
    """
    if not PSUTIL_AVAILABLE:
        # Fallback to a reasonable default if psutil not available
        return min(5000, n_mols)
    
    available_bytes = psutil.virtual_memory().available * memory_usage_fraction
    
    # Bytes per pair: 4 for each uint32 index + 4 for each float32 similarity
    if storage_mode == "both":
        bytes_per_pair = 16  # mol1_idx + mol2_idx + tanimoto + cosine
    else:
        bytes_per_pair = 12  # mol1_idx + mol2_idx + similarity
    
    # For a d x d square chunk storing lower triangular: d*(d+1)/2 pairs maximum
    # Memory needed: d*(d+1)/2 * bytes_per_pair <= available_bytes
    # Approximate: d^2 * bytes_per_pair / 2 <= available_bytes
    # d <= sqrt(2 * available_bytes / bytes_per_pair)
    
    max_d = int((2 * available_bytes / bytes_per_pair) ** 0.5)
    
    # Ensure at least 1, and cap at n_mols
    return max(1, min(max_d, n_mols))


def _get_similarity_func(metric: str):
    """Get the appropriate similarity function based on metric name."""
    if metric == "tanimoto":
        return crossTanimotoSimilarityMemoryConstrained
    elif metric == "cosine":
        return crossCosineSimilarityMemoryConstrained
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def _write_similarity_chunk_to_parquet(
    sim_matrix_chunk: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    parquet_writer: pq.ParquetWriter,
    similarity_column: str = "similarity",
    threshold: Optional[float] = None,
) -> int:
    """
    Write a chunk of similarity matrix to parquet, storing only lower triangular (i <= j).
    
    Args:
        sim_matrix_chunk: Similarity matrix chunk (n_rows x n_cols)
        row_indices: Row indices for this chunk (corresponds to sim_matrix_chunk rows)
        col_indices: Column indices corresponding to sim_matrix_chunk columns
        parquet_writer: PyArrow ParquetWriter instance
        similarity_column: Name of the similarity column ("tanimoto" or "cosine")
        threshold: Optional minimum similarity threshold (only store >= threshold)
        
    Returns:
        Number of rows written
    """
    n_rows, n_cols = sim_matrix_chunk.shape
    
    # Build coordinate arrays
    row_indices = np.asarray(row_indices, dtype=np.uint32)
    
    # For each row, get columns where col_idx >= row_idx (lower triangular)
    mol1_list = []
    mol2_list = []
    similarity_list = []
    
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
            similarity_list.extend(row_vals.tolist())
    
    if not mol1_list:
        return 0
    
    # Create PyArrow table
    table = pa.table({
        "mol1_idx": pa.array(mol1_list, type=pa.uint32()),
        "mol2_idx": pa.array(mol2_list, type=pa.uint32()),
        similarity_column: pa.array(similarity_list, type=pa.float32()),
    })
    
    parquet_writer.write_table(table)
    
    return len(mol1_list)


def _write_similarity_chunk_to_parquet_offdiagonal(
    sim_matrix_chunk: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    parquet_writer: pq.ParquetWriter,
    similarity_column: str = "similarity",
    threshold: Optional[float] = None,
) -> int:
    """
    Write an off-diagonal chunk of similarity matrix to parquet.
    
    For off-diagonal blocks (col_start > row_start), all pairs are valid
    since col_idx > row_idx always holds.
    
    Args:
        sim_matrix_chunk: Similarity matrix chunk (n_rows x n_cols)
        row_indices: Row indices for this chunk
        col_indices: Column indices for this chunk
        parquet_writer: PyArrow ParquetWriter instance
        similarity_column: Name of the similarity column
        threshold: Optional minimum similarity threshold
        
    Returns:
        Number of rows written
    """
    n_rows, n_cols = sim_matrix_chunk.shape
    
    # Build coordinate arrays
    row_indices = np.asarray(row_indices, dtype=np.uint32)
    col_indices = np.asarray(col_indices, dtype=np.uint32)
    
    # For off-diagonal, all pairs are valid (col_idx > row_idx)
    mol1_list = []
    mol2_list = []
    similarity_list = []
    
    for i, row_idx in enumerate(row_indices):
        row_vals = sim_matrix_chunk[i, :]
        col_vals = col_indices
        
        # Apply threshold filtering
        if threshold is not None:
            thresh_mask = row_vals >= threshold
            row_vals = row_vals[thresh_mask]
            col_vals = col_vals[thresh_mask]
        
        if len(row_vals) > 0:
            mol1_list.extend([row_idx] * len(row_vals))
            mol2_list.extend(col_vals.tolist())
            similarity_list.extend(row_vals.tolist())
    
    if not mol1_list:
        return 0
    
    # Create PyArrow table
    table = pa.table({
        "mol1_idx": pa.array(mol1_list, type=pa.uint32()),
        "mol2_idx": pa.array(mol2_list, type=pa.uint32()),
        similarity_column: pa.array(similarity_list, type=pa.float32()),
    })
    
    parquet_writer.write_table(table)
    
    return len(mol1_list)


def _compute_similarity_matrix_chunked_parquet(
    fps_tensor,
    indices: np.ndarray,
    parquet_path: Union[str, Path],
    similarity_metric: str = "tanimoto",
    log_path: Optional[Union[str, Path]] = None,
    memory_usage_fraction: float = 0.5,
    threshold: Optional[float] = None,
) -> int:
    """
    Compute similarity matrix in chunks and write to parquet file.
    Stores only lower triangular (i <= j) with optional threshold filtering.
    
    Args:
        fps_tensor: torch.Tensor of fingerprints on GPU
        indices: Original indices of valid molecules
        parquet_path: Path to output parquet file
        similarity_metric: "tanimoto" or "cosine"
        log_path: Path to log file
        memory_usage_fraction: Fraction of available memory to use
        threshold: Minimum similarity threshold
        
    Returns:
        Total number of similarity pairs written
    """
    n_mols = len(indices)
    total_written = 0
    
    # Calculate square chunk size based on memory (single similarity mode)
    chunk_size = _calculate_square_chunk_size(n_mols, memory_usage_fraction, storage_mode="single")
    n_row_blocks = math.ceil(n_mols / chunk_size)
    
    if log_path:
        _log_message_to_file(
            f"Computing {similarity_metric} similarity matrix with square chunks "
            f"(chunk_size={chunk_size}x{chunk_size}, n_mols={n_mols}, memory_fraction={memory_usage_fraction})...", 
            log_path
        )
        if threshold:
            _log_message_to_file(f"Applying similarity threshold: {threshold}", log_path)
    
    # Get the appropriate similarity function
    similarity_func = _get_similarity_func(similarity_metric)
    
    # Define parquet schema with appropriate column name
    similarity_column = similarity_metric  # "tanimoto" or "cosine"
    schema = pa.schema([
        ("mol1_idx", pa.uint32()),
        ("mol2_idx", pa.uint32()),
        (similarity_column, pa.float32()),
    ])
    
    # Open parquet writer with snappy compression
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunk_counter = 0
    with pq.ParquetWriter(
        str(parquet_path),
        schema,
        compression="snappy",
        use_dictionary=["mol1_idx", "mol2_idx"],
    ) as writer:
        
        # Process row blocks
        for row_start in range(0, n_mols, chunk_size):
            row_end = min(row_start + chunk_size, n_mols)
            
            # For lower triangular, only process column blocks from row_start onwards
            for col_start in range(row_start, n_mols, chunk_size):
                col_end = min(col_start + chunk_size, n_mols)
                chunk_counter += 1
                chunk_start_time = perf_counter()
                
                # Get fingerprint chunks
                row_fps_chunk = fps_tensor[row_start:row_end]
                col_fps_chunk = fps_tensor[col_start:col_end]
                
                # Compute similarity: row_chunk x col_chunk (square submatrix)
                sim_chunk = similarity_func(row_fps_chunk, col_fps_chunk)
                sim_chunk = np.asarray(sim_chunk, dtype=np.float32)
                
                # For diagonal blocks (where row_start == col_start), we need lower triangular
                # For off-diagonal blocks (col_start > row_start), we need all pairs
                if col_start == row_start:
                    # Diagonal block: apply lower triangular filtering
                    rows_written = _write_similarity_chunk_to_parquet(
                        sim_chunk,
                        indices[row_start:row_end],
                        indices[col_start:col_end],
                        writer,
                        similarity_column=similarity_column,
                        threshold=threshold,
                    )
                else:
                    # Off-diagonal block: all pairs are valid (col_idx > row_idx)
                    rows_written = _write_similarity_chunk_to_parquet_offdiagonal(
                        sim_chunk,
                        indices[row_start:row_end],
                        indices[col_start:col_end],
                        writer,
                        similarity_column=similarity_column,
                        threshold=threshold,
                    )
                total_written += rows_written
                
                chunk_time = perf_counter() - chunk_start_time
                if log_path:
                    _log_message_to_file(
                        f"Block {chunk_counter}: rows {row_start}-{row_end-1} x cols {col_start}-{col_end-1}, "
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


def _compute_both_similarities_chunked_parquet(
    fps_tensor,
    indices: np.ndarray,
    parquet_path: Union[str, Path],
    log_path: Optional[Union[str, Path]] = None,
    memory_usage_fraction: float = 0.5,
    threshold: Optional[float] = None,
) -> int:
    """
    Compute both Tanimoto and Cosine similarity matrices in chunks and write to single parquet file.
    Stores only lower triangular (i <= j) with optional threshold filtering.
    Both similarities are calculated from the same fingerprints.
    
    Args:
        fps_tensor: torch.Tensor of fingerprints on GPU
        indices: Original indices of valid molecules
        parquet_path: Path to output parquet file
        log_path: Path to log file
        memory_usage_fraction: Fraction of available memory to use
        threshold: Minimum similarity threshold (applied to both metrics)
        
    Returns:
        Total number of similarity pairs written
    """
    n_mols = len(indices)
    total_written = 0
    
    # Calculate square chunk size based on memory (need memory for both matrices simultaneously)
    chunk_size = _calculate_square_chunk_size(n_mols, memory_usage_fraction / 2, storage_mode="both")
    
    if log_path:
        _log_message_to_file(
            f"Computing BOTH tanimoto and cosine similarity matrices with square chunks "
            f"(chunk_size={chunk_size}x{chunk_size}, n_mols={n_mols}, memory_fraction={memory_usage_fraction})...", 
            log_path
        )
        if threshold:
            _log_message_to_file(f"Applying similarity threshold: {threshold}", log_path)
    
    # Define parquet schema with both similarity columns
    schema = pa.schema([
        ("mol1_idx", pa.uint32()),
        ("mol2_idx", pa.uint32()),
        ("tanimoto", pa.float32()),
        ("cosine", pa.float32()),
    ])
    
    # Open parquet writer with snappy compression
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunk_counter = 0
    with pq.ParquetWriter(
        str(parquet_path),
        schema,
        compression="snappy",
        use_dictionary=["mol1_idx", "mol2_idx"],
    ) as writer:
        
        # Process row blocks
        for row_start in range(0, n_mols, chunk_size):
            row_end = min(row_start + chunk_size, n_mols)
            
            # For lower triangular, only process column blocks from row_start onwards
            for col_start in range(row_start, n_mols, chunk_size):
                col_end = min(col_start + chunk_size, n_mols)
                chunk_counter += 1
                chunk_start_time = perf_counter()
                
                # Get fingerprint chunks
                row_fps_chunk = fps_tensor[row_start:row_end]
                col_fps_chunk = fps_tensor[col_start:col_end]
                
                # Compute both similarities: row_chunk x col_chunk (square submatrix)
                tanimoto_chunk = crossTanimotoSimilarityMemoryConstrained(row_fps_chunk, col_fps_chunk)
                tanimoto_chunk = np.asarray(tanimoto_chunk, dtype=np.float32)
                
                cosine_chunk = crossCosineSimilarityMemoryConstrained(row_fps_chunk, col_fps_chunk)
                cosine_chunk = np.asarray(cosine_chunk, dtype=np.float32)
                
                # For diagonal blocks (where row_start == col_start), we need lower triangular
                # For off-diagonal blocks (col_start > row_start), we need all pairs
                if col_start == row_start:
                    # Diagonal block: apply lower triangular filtering
                    rows_written = _write_both_similarities_chunk_to_parquet(
                        tanimoto_chunk,
                        cosine_chunk,
                        indices[row_start:row_end],
                        indices[col_start:col_end],
                        writer,
                        threshold=threshold,
                    )
                else:
                    # Off-diagonal block: all pairs are valid (col_idx > row_idx)
                    rows_written = _write_both_similarities_chunk_to_parquet_offdiagonal(
                        tanimoto_chunk,
                        cosine_chunk,
                        indices[row_start:row_end],
                        indices[col_start:col_end],
                        writer,
                        threshold=threshold,
                    )
                total_written += rows_written
                
                chunk_time = perf_counter() - chunk_start_time
                if log_path:
                    _log_message_to_file(
                        f"Block {chunk_counter}: rows {row_start}-{row_end-1} x cols {col_start}-{col_end-1}, "
                        f"wrote {rows_written} pairs, "
                        f"took {chunk_time:.2f}s",
                        log_path
                    )
                
                # Cleanup
                del tanimoto_chunk, cosine_chunk
                gc.collect()
    
    if log_path:
        _log_message_to_file(
            f"Parquet writing complete. Total pairs written: {total_written}",
            log_path
        )
    
    return total_written


def _write_both_similarities_chunk_to_parquet(
    tanimoto_chunk: np.ndarray,
    cosine_chunk: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    parquet_writer: pq.ParquetWriter,
    threshold: Optional[float] = None,
) -> int:
    """
    Write a chunk of both similarity matrices to parquet, storing only lower triangular (i <= j).
    
    Args:
        tanimoto_chunk: Tanimoto similarity matrix chunk (n_rows x n_cols)
        cosine_chunk: Cosine similarity matrix chunk (n_rows x n_cols)
        row_indices: Row indices for this chunk (corresponds to chunk rows)
        col_indices: Column indices corresponding to similarity chunks columns
        parquet_writer: PyArrow ParquetWriter instance
        threshold: Optional minimum similarity threshold (applied to tanimoto)
        
    Returns:
        Number of rows written
    """
    n_rows, n_cols = tanimoto_chunk.shape
    
    # Build coordinate arrays
    row_indices = np.asarray(row_indices, dtype=np.uint32)
    
    # For each row, get columns where col_idx >= row_idx (lower triangular)
    mol1_list = []
    mol2_list = []
    tanimoto_list = []
    cosine_list = []
    
    for i, row_idx in enumerate(row_indices):
        # Get valid column indices for this row (lower triangular only: j >= i)
        valid_mask = col_indices >= row_idx
        if not np.any(valid_mask):
            continue
            
        tanimoto_vals = tanimoto_chunk[i, valid_mask]
        cosine_vals = cosine_chunk[i, valid_mask]
        col_vals = col_indices[valid_mask]
        
        # Apply threshold filtering (based on tanimoto)
        if threshold is not None:
            thresh_mask = tanimoto_vals >= threshold
            tanimoto_vals = tanimoto_vals[thresh_mask]
            cosine_vals = cosine_vals[thresh_mask]
            col_vals = col_vals[thresh_mask]
        
        if len(tanimoto_vals) > 0:
            mol1_list.extend([row_idx] * len(tanimoto_vals))
            mol2_list.extend(col_vals.tolist())
            tanimoto_list.extend(tanimoto_vals.tolist())
            cosine_list.extend(cosine_vals.tolist())
    
    if not mol1_list:
        return 0
    
    # Create PyArrow table
    table = pa.table({
        "mol1_idx": pa.array(mol1_list, type=pa.uint32()),
        "mol2_idx": pa.array(mol2_list, type=pa.uint32()),
        "tanimoto": pa.array(tanimoto_list, type=pa.float32()),
        "cosine": pa.array(cosine_list, type=pa.float32()),
    })
    
    parquet_writer.write_table(table)
    
    return len(mol1_list)


def _write_both_similarities_chunk_to_parquet_offdiagonal(
    tanimoto_chunk: np.ndarray,
    cosine_chunk: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    parquet_writer: pq.ParquetWriter,
    threshold: Optional[float] = None,
) -> int:
    """
    Write an off-diagonal chunk of both similarity matrices to parquet.
    
    For off-diagonal blocks (col_start > row_start), all pairs are valid
    since col_idx > row_idx always holds.
    
    Args:
        tanimoto_chunk: Tanimoto similarity matrix chunk (n_rows x n_cols)
        cosine_chunk: Cosine similarity matrix chunk (n_rows x n_cols)
        row_indices: Row indices for this chunk
        col_indices: Column indices for this chunk
        parquet_writer: PyArrow ParquetWriter instance
        threshold: Optional minimum similarity threshold (applied to tanimoto)
        
    Returns:
        Number of rows written
    """
    n_rows, n_cols = tanimoto_chunk.shape
    
    # Build coordinate arrays
    row_indices = np.asarray(row_indices, dtype=np.uint32)
    col_indices = np.asarray(col_indices, dtype=np.uint32)
    
    # For off-diagonal, all pairs are valid (col_idx > row_idx)
    mol1_list = []
    mol2_list = []
    tanimoto_list = []
    cosine_list = []
    
    for i, row_idx in enumerate(row_indices):
        tanimoto_vals = tanimoto_chunk[i, :]
        cosine_vals = cosine_chunk[i, :]
        col_vals = col_indices
        
        # Apply threshold filtering (based on tanimoto)
        if threshold is not None:
            thresh_mask = tanimoto_vals >= threshold
            tanimoto_vals = tanimoto_vals[thresh_mask]
            cosine_vals = cosine_vals[thresh_mask]
            col_vals = col_vals[thresh_mask]
        
        if len(tanimoto_vals) > 0:
            mol1_list.extend([row_idx] * len(tanimoto_vals))
            mol2_list.extend(col_vals.tolist())
            tanimoto_list.extend(tanimoto_vals.tolist())
            cosine_list.extend(cosine_vals.tolist())
    
    if not mol1_list:
        return 0
    
    # Create PyArrow table
    table = pa.table({
        "mol1_idx": pa.array(mol1_list, type=pa.uint32()),
        "mol2_idx": pa.array(mol2_list, type=pa.uint32()),
        "tanimoto": pa.array(tanimoto_list, type=pa.float32()),
        "cosine": pa.array(cosine_list, type=pa.float32()),
    })
    
    parquet_writer.write_table(table)
    
    return len(mol1_list)


def calculate_similarity_matrix(
    smiles: List[str],
    parquet_path: Union[str, Path],
    indices: Optional[np.ndarray] = None,
    fp_params: Optional[FingerprintParams] = None,
    similarity_metric: Literal["tanimoto", "cosine", "both"] = "tanimoto",
    threshold: Optional[float] = None,
    memory_usage_fraction: float = 0.5,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Calculate similarity matrix for a list of SMILES strings using GPU.
    
    Writes lower-triangular similarities to parquet file (memory efficient).
    
    Args:
        smiles: List of SMILES strings.
        parquet_path: Path to output parquet file (required).
        indices: Original indices for molecules (caller-provided). If None, uses range(len(smiles)).
        fp_params: Fingerprint parameters (FingerprintParams dataclass). Defaults to Morgan r=2, size=2048.
        similarity_metric: "tanimoto", "cosine", or "both".
        threshold: Minimum similarity threshold (only store similarities >= threshold).
        memory_usage_fraction: Fraction of available CPU memory to use (0.0-1.0, default 0.5).
        log_path: Path to log progress. If None, uses parquet_path with .log suffix.
        
    Raises:
        ValueError: If invalid similarity_metric.
        ImportError: If nvmolkit is not available for GPU acceleration.
    """
    if not NVMOLKIT_AVAILABLE:
        raise ImportError("nvmolkit is required for GPU similarity calculation")
    
    if similarity_metric not in ("tanimoto", "cosine", "both"):
        raise ValueError(f"similarity_metric must be 'tanimoto', 'cosine', or 'both', got '{similarity_metric}'")
    
    # Auto-generate log path if not provided
    if log_path is None:
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
            f"Starting {similarity_metric} similarity matrix calculation for {len(smiles)} molecules.",
            log_path,
            overwrite=True
        )
    
    try:
        # Generate fingerprints on GPU
        fps_tensor, valid_indices, valid_smiles = _generate_fingerprints_gpu(
            smiles, fp_params, log_path
        )
        
        # Filter indices to match valid molecules
        valid_indices_arr = np.array(valid_indices, dtype=np.uint32)
        indices = indices[valid_indices_arr]
        
        # Compute similarities in chunks and write to parquet
        if similarity_metric == "both":
            _compute_both_similarities_chunked_parquet(
                fps_tensor,
                indices,
                parquet_path,
                log_path=log_path,
                memory_usage_fraction=memory_usage_fraction,
                threshold=threshold,
            )
        else:
            _compute_similarity_matrix_chunked_parquet(
                fps_tensor,
                indices,
                parquet_path,
                similarity_metric=similarity_metric,
                log_path=log_path,
                memory_usage_fraction=memory_usage_fraction,
                threshold=threshold,
            )
        
        end = perf_counter()
        if log_path:
            _log_message_to_file(
                f"Parquet calculation completed in {end - start:.2f}s. "
                f"Output: {parquet_path}",
                log_path
            )
            
    except Exception:
        err = traceback.format_exc()
        if log_path:
            _log_message_to_file(f"Error in matrix calculation:\n{err}", log_path, level=logging.ERROR)
        raise


def calculate_similarity_matrix_streaming(
    parquet_path: Union[str, Path],
    output_parquet_path: Union[str, Path],
    smiles_column: str = "smiles",
    index_column: str = "index",
    fp_params: Optional[FingerprintParams] = None,
    similarity_metric: Literal["tanimoto", "cosine", "both"] = "tanimoto",
    threshold: Optional[float] = None,
    memory_usage_fraction: float = 0.5,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Calculate similarity matrix by reading SMILES from a parquet file using polars streaming.
    
    Args:
        parquet_path: Path to input parquet file containing SMILES.
        output_parquet_path: Path for output parquet file (required).
        smiles_column: Name of column containing SMILES strings.
        index_column: Name of column containing molecule indices.
        fp_params: Fingerprint parameters.
        similarity_metric: "tanimoto", "cosine", or "both".
        threshold: Minimum similarity threshold.
        memory_usage_fraction: Fraction of available CPU memory to use.
        log_path: Path to log file.
    """
    if log_path is None:
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
    
    return calculate_similarity_matrix(
        smiles=smiles_list,
        parquet_path=output_parquet_path,
        indices=indices,
        fp_params=fp_params,
        similarity_metric=similarity_metric,
        threshold=threshold,
        memory_usage_fraction=memory_usage_fraction,
        log_path=log_path,
    )


# Backward compatibility aliases
def calculate_tanimoto_matrix(
    smiles: List[str],
    parquet_path: Union[str, Path],
    indices: Optional[np.ndarray] = None,
    fp_params: Optional[FingerprintParams] = None,
    threshold: Optional[float] = None,
    memory_usage_fraction: float = 0.5,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Calculate Tanimoto similarity matrix (backward compatibility alias).
    
    This is an alias for calculate_similarity_matrix with similarity_metric="tanimoto".
    """
    return calculate_similarity_matrix(
        smiles=smiles,
        parquet_path=parquet_path,
        indices=indices,
        fp_params=fp_params,
        similarity_metric="tanimoto",
        threshold=threshold,
        memory_usage_fraction=memory_usage_fraction,
        log_path=log_path,
    )


def calculate_cosine_matrix(
    smiles: List[str],
    parquet_path: Union[str, Path],
    indices: Optional[np.ndarray] = None,
    fp_params: Optional[FingerprintParams] = None,
    threshold: Optional[float] = None,
    memory_usage_fraction: float = 0.5,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Calculate Cosine similarity matrix.
    
    This is an alias for calculate_similarity_matrix with similarity_metric="cosine".
    """
    return calculate_similarity_matrix(
        smiles=smiles,
        parquet_path=parquet_path,
        indices=indices,
        fp_params=fp_params,
        similarity_metric="cosine",
        threshold=threshold,
        memory_usage_fraction=memory_usage_fraction,
        log_path=log_path,
    )


# Backward compatibility for streaming functions
def calculate_tanimoto_matrix_streaming(
    parquet_path: Union[str, Path],
    output_parquet_path: Union[str, Path],
    smiles_column: str = "smiles",
    index_column: str = "index",
    fp_params: Optional[FingerprintParams] = None,
    threshold: Optional[float] = None,
    memory_usage_fraction: float = 0.5,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Calculate Tanimoto matrix by reading SMILES from parquet (backward compatibility alias).
    """
    return calculate_similarity_matrix_streaming(
        parquet_path=parquet_path,
        output_parquet_path=output_parquet_path,
        smiles_column=smiles_column,
        index_column=index_column,
        fp_params=fp_params,
        similarity_metric="tanimoto",
        threshold=threshold,
        memory_usage_fraction=memory_usage_fraction,
        log_path=log_path,
    )


def calculate_cosine_matrix_streaming(
    parquet_path: Union[str, Path],
    output_parquet_path: Union[str, Path],
    smiles_column: str = "smiles",
    index_column: str = "index",
    fp_params: Optional[FingerprintParams] = None,
    threshold: Optional[float] = None,
    memory_usage_fraction: float = 0.5,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Calculate Cosine matrix by reading SMILES from parquet.
    """
    return calculate_similarity_matrix_streaming(
        parquet_path=parquet_path,
        output_parquet_path=output_parquet_path,
        smiles_column=smiles_column,
        index_column=index_column,
        fp_params=fp_params,
        similarity_metric="cosine",
        threshold=threshold,
        memory_usage_fraction=memory_usage_fraction,
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
