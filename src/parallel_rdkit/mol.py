from typing import List, Iterable
from .parallel_rdkit_backend import sanitize_smiles_parallel

def sanitize_smiles(smiles: Iterable[str], batch_size: int = 1000) -> List[str]:
    """
    Sanitize a list of SMILES strings in parallel.
    
    Args:
        smiles: Iterable of SMILES strings.
        batch_size: Ignored in this implementation as C++ handles batching.
        
    Returns:
        List of sanitized SMILES strings.
    """
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return sanitize_smiles_parallel(smiles)
