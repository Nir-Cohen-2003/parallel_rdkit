from typing import List, Optional, Union
from pathlib import Path

import numpy as np

from .parallel_rdkit_backend import (
    screen_smarts_direct as _screen_smarts_direct,
    screen_smarts_streaming as _screen_smarts_streaming,
)


def screen_smarts(
    smarts_list: List[str],
    smiles_file: Union[str, Path],
    mode: str = "direct",
    batch_size: int = 10000,
    cache_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Union[np.ndarray, int]:
    """
    Screen molecules from a SMILES file against a list of SMARTS patterns.

    Parameters
    ----------
    smarts_list : List[str]
        List of SMARTS patterns to screen against.
    smiles_file : str or Path
        Path to file containing SMILES (one per line).
    mode : str, default "direct"
        Either "direct" (load all into memory) or "streaming" (process in batches).
    batch_size : int, default 10000
        Number of molecules to process per batch (streaming mode only).
    cache_path : str or Path, optional
        Path to cache file. If provided and cache is valid, skips computation.
        For streaming mode, this caches only metadata (hash, params).
    output_path : str or Path, optional
        Required for streaming mode. Path to output .npy file.

    Returns
    -------
    np.ndarray or int
        For direct mode: N x M boolean array (N molecules, M SMARTS patterns).
        For streaming mode: number of molecules processed.

    Examples
    --------
    >>> # Direct mode - returns numpy array
    >>> result = screen_smarts(
    ...     smarts_list=["[OH]", "[NH2]", "c1ccccc1"],
    ...     smiles_file="molecules.smi",
    ...     mode="direct",
    ...     cache_path="cache.bin"
    ... )
    >>> result.shape
    (1000, 3)  # 1000 molecules, 3 SMARTS patterns

    >>> # Streaming mode - writes to file, returns count
    >>> count = screen_smarts(
    ...     smarts_list=["[OH]", "[NH2]", "c1ccccc1"],
    ...     smiles_file="molecules.smi",
    ...     mode="streaming",
    ...     batch_size=5000,
    ...     output_path="results.npy"
    ... )
    """
    smiles_file = str(smiles_file)
    cache_str = str(cache_path) if cache_path else ""
    
    if mode == "direct":
        result = _screen_smarts_direct(smiles_file, smarts_list, cache_str)
        # Convert to numpy array
        return np.array(result, dtype=bool)
    
    elif mode == "streaming":
        if output_path is None:
            raise ValueError("output_path is required for streaming mode")
        
        output_str = str(output_path)
        if not output_str.endswith('.npy'):
            output_str += '.npy'
        
        count = _screen_smarts_streaming(
            smiles_file, 
            smarts_list, 
            batch_size, 
            cache_str,
            output_str
        )
        return count
    
    else:
        raise ValueError(f"mode must be 'direct' or 'streaming', got {mode}")
