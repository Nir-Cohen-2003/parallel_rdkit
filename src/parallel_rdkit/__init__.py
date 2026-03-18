import os
from multiprocessing import get_context, cpu_count
from typing import Iterable, List, Sequence, Tuple

from .fingerprint import (
    FingerprintParams,
    get_fp_list,
)
from .screen_smarts import (
    screen_smarts,
)

# Import similarity matrix functions (GPU-accelerated, requires nvmolkit)
try:
    from .matrix_similarity import (
        calculate_similarity_matrix,
        calculate_similarity_matrix_streaming,
        calculate_tanimoto_matrix,
        calculate_cosine_matrix,
        calculate_tanimoto_matrix_streaming,
        calculate_cosine_matrix_streaming,
        butina_split,
        umap_split,
    )
    _MATRIX_SIMILARITY_AVAILABLE = True
except ImportError:
    _MATRIX_SIMILARITY_AVAILABLE = False

__all__ = [
    "msready_smiles",
    "msready_smiles_parallel",
    "sanitize_smiles_parallel",
    "inchi_to_smiles_parallel",
    "smiles_to_inchi_parallel",
    "smiles_to_inchikey_parallel",
    "msready_inchi_inchikey_parallel",
    "FingerprintParams",
    "get_fp_list",
    "get_fp_polars",
    "screen_smarts",
    "calculate_similarity_matrix",
    "calculate_similarity_matrix_streaming",
    "calculate_tanimoto_matrix",
    "calculate_cosine_matrix",
    "calculate_tanimoto_matrix_streaming",
    "calculate_cosine_matrix_streaming",
    "butina_split",
    "umap_split",
]


# Constants for multiprocessing
MIN_CHUNK_SIZE = 2500


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# Worker functions for multiprocessing
# These are module-level functions so they can be pickled

def _inchi_to_smiles_worker(chunk: List[str]) -> List[str]:
    """Worker function for InChI to SMILES conversion."""
    from .parallel_rdkit_backend import inchi_to_smiles_parallel as _inchi_to_smiles_parallel
    return _inchi_to_smiles_parallel(chunk)


def _smiles_to_inchi_worker(chunk: List[str]) -> List[str]:
    """Worker function for SMILES to InChI conversion."""
    from .parallel_rdkit_backend import smiles_to_inchi_parallel as _smiles_to_inchi_parallel
    return _smiles_to_inchi_parallel(chunk)


def _smiles_to_inchikey_worker(chunk: List[str]) -> List[str]:
    """Worker function for SMILES to InChIKey conversion."""
    from .parallel_rdkit_backend import smiles_to_inchikey_parallel as _smiles_to_inchikey_parallel
    return _smiles_to_inchikey_parallel(chunk)


def _msready_inchi_inchikey_worker(chunk: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Worker function for MS-Ready + InChI + InChIKey conversion."""
    from .parallel_rdkit_backend import msready_inchi_inchikey_parallel as _msready_inchi_inchikey_parallel
    return _msready_inchi_inchikey_parallel(chunk)


def msready_smiles(smiles: str) -> str:
    """Transforms a SMILES string into an MS-Ready SMILES string."""
    from .parallel_rdkit_backend import msready_smiles as _msready_smiles
    return _msready_smiles(smiles)


def msready_smiles_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel MS-Ready transformation of SMILES."""
    from .parallel_rdkit_backend import msready_smiles_parallel as _msready_smiles_parallel
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _msready_smiles_parallel(smiles)


def sanitize_smiles_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel SMILES sanitization."""
    from .parallel_rdkit_backend import sanitize_smiles_parallel as _sanitize_smiles_parallel
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _sanitize_smiles_parallel(smiles)


def inchi_to_smiles_parallel(inchis: Iterable[str]) -> List[str]:
    """Parallel InChI to SMILES conversion using multiprocessing."""
    if not isinstance(inchis, list):
        inchis = list(inchis)
    
    if not inchis:
        return []
    
    # For small datasets, just process directly without multiprocessing overhead
    if len(inchis) <= MIN_CHUNK_SIZE:
        return _inchi_to_smiles_worker(inchis)
    
    # Split into chunks
    chunks = _chunk_list(inchis, MIN_CHUNK_SIZE)
    
    # Use spawn context for cross-platform compatibility
    ctx = get_context('spawn')
    
    # Determine number of processes (auto-detect based on CPU count)
    num_processes = min(cpu_count(), len(chunks))
    
    with ctx.Pool(processes=num_processes) as pool:
        # Use imap for memory efficiency with large datasets
        results = list(pool.imap(_inchi_to_smiles_worker, chunks))
    
    # Flatten results
    flattened = []
    for chunk_result in results:
        flattened.extend(chunk_result)
    
    return flattened


def smiles_to_inchi_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel SMILES to InChI conversion using multiprocessing."""
    if not isinstance(smiles, list):
        smiles = list(smiles)
    
    if not smiles:
        return []
    
    # For small datasets, just process directly without multiprocessing overhead
    if len(smiles) <= MIN_CHUNK_SIZE:
        return _smiles_to_inchi_worker(smiles)
    
    # Split into chunks
    chunks = _chunk_list(smiles, MIN_CHUNK_SIZE)
    
    # Use spawn context for cross-platform compatibility
    ctx = get_context('spawn')
    
    # Determine number of processes (auto-detect based on CPU count)
    num_processes = min(cpu_count(), len(chunks))
    
    with ctx.Pool(processes=num_processes) as pool:
        # Use imap for memory efficiency with large datasets
        results = list(pool.imap(_smiles_to_inchi_worker, chunks))
    
    # Flatten results
    flattened = []
    for chunk_result in results:
        flattened.extend(chunk_result)
    
    return flattened


def smiles_to_inchikey_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel SMILES to InChIKey conversion using multiprocessing."""
    if not isinstance(smiles, list):
        smiles = list(smiles)
    
    if not smiles:
        return []
    
    # For small datasets, just process directly without multiprocessing overhead
    if len(smiles) <= MIN_CHUNK_SIZE:
        return _smiles_to_inchikey_worker(smiles)
    
    # Split into chunks
    chunks = _chunk_list(smiles, MIN_CHUNK_SIZE)
    
    # Use spawn context for cross-platform compatibility
    ctx = get_context('spawn')
    
    # Determine number of processes (auto-detect based on CPU count)
    num_processes = min(cpu_count(), len(chunks))
    
    with ctx.Pool(processes=num_processes) as pool:
        # Use imap for memory efficiency with large datasets
        results = list(pool.imap(_smiles_to_inchikey_worker, chunks))
    
    # Flatten results
    flattened = []
    for chunk_result in results:
        flattened.extend(chunk_result)
    
    return flattened


def msready_inchi_inchikey_parallel(
    smiles: Iterable[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Parallel conversion to MS-Ready SMILES, InChI, and InChIKey simultaneously.
    Uses multiprocessing for thread-safe INCHI operations.
    Returns a tuple of lists: (list of MS-Ready SMILES, list of InChIs, list of InChIKeys)
    """
    if not isinstance(smiles, list):
        smiles = list(smiles)
    
    if not smiles:
        return ([], [], [])
    
    # For small datasets, process directly
    if len(smiles) <= MIN_CHUNK_SIZE:
        return _msready_inchi_inchikey_worker(smiles)
    
    # Split into chunks
    chunks = _chunk_list(smiles, MIN_CHUNK_SIZE)
    
    # Use spawn context for cross-platform compatibility
    ctx = get_context('spawn')
    
    # Determine number of processes (auto-detect based on CPU count)
    num_processes = min(cpu_count(), len(chunks))
    
    with ctx.Pool(processes=num_processes) as pool:
        # Use imap for memory efficiency with large datasets
        results = list(pool.imap(_msready_inchi_inchikey_worker, chunks))
    
    # Unpack and flatten results
    msready_all = []
    inchi_all = []
    inchikey_all = []
    
    for msready_chunk, inchi_chunk, inchikey_chunk in results:
        msready_all.extend(msready_chunk)
        inchi_all.extend(inchi_chunk)
        inchikey_all.extend(inchikey_chunk)
    
    return (msready_all, inchi_all, inchikey_all)
