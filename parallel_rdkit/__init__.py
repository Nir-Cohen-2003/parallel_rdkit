from typing import List, Sequence, Tuple, Iterable
from .parallel_rdkit_backend import (
    msready_smiles as _msready_smiles,
    msready_smiles_parallel as _msready_smiles_parallel,
    sanitize_smiles_parallel as _sanitize_smiles_parallel,
    inchi_to_smiles_parallel as _inchi_to_smiles_parallel,
    smiles_to_inchi_parallel as _smiles_to_inchi_parallel,
    smiles_to_inchikey_parallel as _smiles_to_inchikey_parallel,
    msready_inchi_inchikey_parallel as _msready_inchi_inchikey_parallel,
)
from .fingerprint import (
    FingerprintParams,
    get_fp_list,
    get_fp_polars,
)

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
]

def msready_smiles(smiles: str) -> str:
    """Transforms a SMILES string into an MS-Ready SMILES string."""
    return _msready_smiles(smiles)

def msready_smiles_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel MS-Ready transformation of SMILES."""
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _msready_smiles_parallel(smiles)

def sanitize_smiles_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel SMILES sanitization."""
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _sanitize_smiles_parallel(smiles)

def inchi_to_smiles_parallel(inchis: Iterable[str]) -> List[str]:
    """Parallel InChI to SMILES conversion."""
    if not isinstance(inchis, list):
        inchis = list(inchis)
    return _inchi_to_smiles_parallel(inchis)

def smiles_to_inchi_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel SMILES to InChI conversion."""
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _smiles_to_inchi_parallel(smiles)

def smiles_to_inchikey_parallel(smiles: Iterable[str]) -> List[str]:
    """Parallel SMILES to InChIKey conversion."""
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _smiles_to_inchikey_parallel(smiles)

def msready_inchi_inchikey_parallel(smiles: Iterable[str]) -> List[Tuple[str, str, str]]:
    """
    Parallel conversion to MS-Ready SMILES, InChI, and InChIKey simultaneously.
    Returns a list of tuples: (MS-Ready SMILES, InChI, InChIKey)
    """
    if not isinstance(smiles, list):
        smiles = list(smiles)
    return _msready_inchi_inchikey_parallel(smiles)
