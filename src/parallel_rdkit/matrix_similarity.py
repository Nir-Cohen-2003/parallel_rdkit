"""Compatibility imports for the generic clustering helpers.

The historical module path remains available for callers that imported the
clustering helpers from ``parallel_rdkit.matrix_similarity``.
"""

from .clustering import butina_split, umap_split

__all__ = ["butina_split", "umap_split"]
