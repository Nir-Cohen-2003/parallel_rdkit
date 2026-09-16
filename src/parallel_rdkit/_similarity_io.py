"""Private NumPy transaction helper for the GPU streaming adapter.

CPU dense file output is implemented by the native extension; this helper is
retained only for the approved GPU Python batch-pair path.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def _write_npy_transaction(path, writer, shape, overwrite):
    """Write a valid .npy sibling, then publish it atomically."""
    path = Path(path)
    parent = path.parent if str(path.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(tmp, "wb") as stream:
            np.lib.format.write_array_header_2_0(stream, {
                "descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
                "fortran_order": False,
                "shape": tuple(int(x) for x in shape),
            })
            data_offset = stream.tell()
            total = int(shape[0]) * int(shape[1]) * 4
            stream.truncate(data_offset + total)
            stream.flush()
            os.fsync(stream.fileno())
        if shape[0] and shape[1]:
            mapped = np.memmap(tmp, mode="r+", dtype=np.float32, shape=shape,
                               order="C", offset=data_offset)
            try:
                writer(mapped)
                mapped.flush()
            finally:
                del mapped
        with open(tmp, "rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


__all__ = ["_write_npy_transaction"]
