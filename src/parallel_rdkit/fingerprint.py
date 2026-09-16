from typing import Iterable, List, Optional, Union, Tuple

import numpy as np

from .parallel_rdkit_backend import FingerprintOptions, get_fingerprints_parallel


class FingerprintParams:
    def __init__(
        self,
        fp_type: str = "morgan",
        fp_method: str = "GetFingerprint",
        fpSize: int = 2048,
        radius: Optional[int] = None,
        useBondTypes: bool = True,
        minPath: int = 1,
        maxPath: int = 7,
        numBitsPerFeature: int = 2,
        use2D: bool = True,
        minDistance: int = 1,
        maxDistance: Optional[int] = None,
        countSimulation: Optional[bool] = None,
        includeChirality: bool = False,
        targetSize: int = 4,
    ):
        self.fp_type = fp_type
        self.fp_method = fp_method
        self.fpSize = fpSize

        # Match RDKit defaults
        if radius is None:
            self.radius = 3 if fp_type == "morgan" else 2
        else:
            self.radius = radius

        if maxDistance is None:
            self.maxDistance = 30  # RDKit default for AtomPair/Torsion is 30
        else:
            self.maxDistance = maxDistance

        if countSimulation is None:
            if fp_type in ["atompair", "torsion"]:
                self.countSimulation = True
            else:
                self.countSimulation = False
        else:
            self.countSimulation = countSimulation

        self.useBondTypes = useBondTypes
        self.minPath = minPath
        self.maxPath = maxPath
        self.numBitsPerFeature = numBitsPerFeature
        self.use2D = use2D
        self.minDistance = minDistance
        self.includeChirality = includeChirality
        self.targetSize = targetSize

    def to_backend_opts(self) -> FingerprintOptions:
        _validate_fingerprint_params(self, allow_count_methods=True)
        opts = FingerprintOptions()
        opts.fp_type = self.fp_type
        opts.fp_method = self.fp_method
        opts.fpSize = self.fpSize
        opts.radius = self.radius
        opts.useBondTypes = self.useBondTypes
        opts.minPath = self.minPath
        opts.maxPath = self.maxPath
        opts.numBitsPerFeature = self.numBitsPerFeature
        opts.use2D = self.use2D
        opts.minDistance = self.minDistance
        opts.maxDistance = self.maxDistance
        opts.countSimulation = self.countSimulation
        opts.includeChirality = self.includeChirality
        opts.targetSize = self.targetSize
        return opts


def _validate_fingerprint_params(params: FingerprintParams, *, allow_count_methods=False) -> None:
    if not isinstance(params.fp_type, str) or params.fp_type not in {
            "morgan", "rdkit", "atompair", "torsion", "maccs"}:
        raise ValueError("unsupported fingerprint family")
    allowed_methods = {"GetFingerprint", "GetSparseFingerprint"}
    if allow_count_methods:
        allowed_methods.update({"GetCountFingerprint", "GetSparseCountFingerprint"})
    if not isinstance(params.fp_method, str) or params.fp_method not in allowed_methods:
        raise ValueError("unsupported fingerprint method")

    int_max = np.iinfo(np.int32).max
    def integer(value, name, minimum):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        value = int(value)
        if value < minimum or value > int_max:
            raise ValueError(f"{name} must be in [{minimum}, {int_max}]")

    def boolean(value, name):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean")

    integer(params.fpSize, "fpSize", 1)
    integer(params.radius, "radius", 0)
    integer(params.minPath, "minPath", 1)
    integer(params.maxPath, "maxPath", 1)
    integer(params.numBitsPerFeature, "numBitsPerFeature", 1)
    integer(params.minDistance, "minDistance", 1)
    integer(params.maxDistance, "maxDistance", 1)
    integer(params.targetSize, "targetSize", 1)
    if int(params.minPath) > int(params.maxPath):
        raise ValueError("minPath must not exceed maxPath")
    if int(params.minDistance) > int(params.maxDistance):
        raise ValueError("minDistance must not exceed maxDistance")
    for name in ("useBondTypes", "use2D", "countSimulation", "includeChirality"):
        boolean(getattr(params, name), name)


def get_fp_list(smiles: Iterable[str], params: FingerprintParams, return_numpy: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[bool]]]:
    """
    Get fingerprints for a list of SMILES strings.
    
    Args:
        smiles: Iterable of SMILES strings.
        params: Fingerprint parameters.
        return_numpy: If True (default), returns a tuple of (2D numpy array, 1D boolean array). 
                      If False, returns a tuple of (list of 1D numpy arrays, list of bools).
                      
    Returns:
        A tuple (fingerprints, valid_mask).
    """
    if not isinstance(smiles, list):
        smiles = list(smiles)

    # C++ returns a tuple (flattened float vector, valid bool vector)
    flattened, valid = get_fingerprints_parallel(smiles, params.to_backend_opts())

    n = len(smiles)
    stride = params.fpSize

    # Reshape and convert to numpy arrays
    arr = np.array(flattened, dtype=np.float32).reshape(n, stride)
    valid_arr = np.array(valid, dtype=bool)

    if params.fp_type == "maccs":
        arr = arr[:, :167]

    if return_numpy:
        return arr, valid_arr
    
    return [arr[i] for i in range(n)], valid_arr.tolist()
