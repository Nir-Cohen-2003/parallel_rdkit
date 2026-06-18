from typing import Iterable, List, Union

import numpy as np
import polars as pl

from .parallel_rdkit_backend import (
    sanitize_smiles_parallel,
    smiles_to_formula_parallel as _smiles_to_formula_parallel,
)


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


# Element order for molecular formula arrays.
FORMULA_ELEMENT_ORDER = ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"]
_NUM_FORMULA_ELEMENTS = len(FORMULA_ELEMENT_ORDER)


def smiles_to_formula(smiles: Union[Iterable[str], pl.Series]) -> Union[np.ndarray, pl.Series]:
    """
    Compute molecular formulas for a list or polars Series of SMILES strings.

    The heavy lifting runs in C++ and is parallelized over molecules using
    OpenMP. The C++ backend returns a flattened (n*12) array of int64 counts
    which is reshaped to (n, 12) in Python.

    Element counts are returned in the fixed order:
    ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"]

    The return type depends on the input type:
    - If ``smiles`` is a ``polars.Series``, the 2D numpy array is ingested
      directly into a ``polars.Series`` of dtype ``Array(Int64, 12)`` without
      creating a Python list of arrays.
    - Otherwise, a 2D ``int64`` numpy array of shape ``(n, 12)`` is returned.

    Invalid SMILES produce a row of zeros. Atoms whose elements are not in the
    12-element list above are silently ignored.

    Args:
        smiles: An iterable of SMILES strings, or a polars Series of SMILES.

    Returns:
        ``np.ndarray`` of shape ``(n, 12)`` for list/iterable input, or a
        ``pl.Series`` of dtype ``Array(Int64, 12)`` for polars Series input.

    Example (list input):
        >>> from parallel_rdkit import smiles_to_formula
        >>> smiles_to_formula(["CCO", "CF"])
        array([[6, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)

    Example (polars input):
        >>> import polars as pl
        >>> from parallel_rdkit import smiles_to_formula
        >>> s = pl.Series(["CCO", "CF"])
        >>> smiles_to_formula(s)
        shape: (2,)
        Series: 'formula' [array[i64, 12]]
        [
            [6, 2, 0, 1, 0, 0, ... 0]
            [3, 1, 0, 0, 1, 0, ... 0]
        ]
    """
    return_polars = isinstance(smiles, pl.Series)
    if return_polars:
        smiles_list = smiles.to_list()
    else:
        if not isinstance(smiles, list):
            smiles = list(smiles)
        smiles_list = smiles

    n = len(smiles_list)
    flattened = _smiles_to_formula_parallel(smiles_list)
    arr = np.array(flattened, dtype=np.int64).reshape(n, _NUM_FORMULA_ELEMENTS)

    if return_polars:
        return pl.Series("formula", arr, dtype=pl.Array(pl.Int64, _NUM_FORMULA_ELEMENTS))
    return arr
