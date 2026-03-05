from typing import Iterable, List, Optional

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


def get_fp_list(smiles: Iterable[str], params: FingerprintParams) -> List[np.ndarray]:
    """
    Get a list of fingerprints for a list of SMILES strings.
    """
    if not isinstance(smiles, list):
        smiles = list(smiles)

    # C++ returns a flattened float vector
    flattened = get_fingerprints_parallel(smiles, params.to_backend_opts())

    n = len(smiles)
    stride = params.fpSize

    # Reshape and convert to numpy arrays
    arr = np.array(flattened, dtype=np.float32).reshape(n, stride)

    if params.fp_type == "maccs":
        arr = arr[:, :167]

    return [arr[i] for i in range(n)]
