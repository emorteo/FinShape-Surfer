"""Read the legacy MXNet ndarray container used by FinFindR checkpoints.

This parser is intentionally narrow. FinFindR 0.1.21 stores float32 CPU
arrays in MXNet's legacy binary format; unsupported storage or data types are
rejected instead of being guessed.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO, Dict, Tuple

import numpy as np

from .errors import FinFindRModelError

_NDARRAY_V2 = 0xF993FAC9
_NDARRAY_V1 = 0xF993FAC8
_MAX_ARRAYS = 100_000
_MAX_DIMENSIONS = 16
_MAX_ELEMENTS = 500_000_000


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise FinFindRModelError("Unexpected end of MXNet parameter file.")
    return value


def _unpack(stream: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack("<" + fmt, _read_exact(stream, size))


def _shape(stream: BinaryIO, first_word: int) -> Tuple[int, ...]:
    if first_word == _NDARRAY_V2:
        (storage_type,) = _unpack(stream, "i")
        if storage_type != 0:
            raise FinFindRModelError(
                f"Unsupported MXNet storage type {storage_type}; only dense arrays are accepted."
            )
        (ndim,) = _unpack(stream, "I")
        dimension_format = "q"
    elif first_word == _NDARRAY_V1:
        (ndim,) = _unpack(stream, "I")
        dimension_format = "q"
    else:
        ndim = first_word
        dimension_format = "I"

    if ndim > _MAX_DIMENSIONS:
        raise FinFindRModelError(f"Unreasonable MXNet array rank: {ndim}.")
    dimensions = tuple(int(value) for value in _unpack(stream, dimension_format * ndim))
    if any(value < 0 for value in dimensions):
        raise FinFindRModelError(f"Negative MXNet array dimension: {dimensions!r}.")
    elements = int(np.prod(dimensions, dtype=np.int64)) if dimensions else 1
    if elements > _MAX_ELEMENTS:
        raise FinFindRModelError(f"MXNet array is too large to load safely: {elements} values.")
    return dimensions


def read_mxnet_params(path: Path) -> Dict[str, np.ndarray]:
    """Return checkpoint arrays keyed by MXNet symbol input name."""

    path = Path(path)
    try:
        stream = io.BytesIO(path.read_bytes())
    except OSError as exc:
        raise FinFindRModelError(f"Cannot read FinFindR model parameters: {path}") from exc

    _header, _reserved, count = _unpack(stream, "QQQ")
    if count > _MAX_ARRAYS:
        raise FinFindRModelError(f"Unreasonable MXNet parameter count: {count}.")

    arrays = []
    for _ in range(count):
        (first_word,) = _unpack(stream, "I")
        shape = _shape(stream, first_word)
        dev_type, _dev_id, type_flag = _unpack(stream, "iii")
        if dev_type not in (1, 2, 3):
            raise FinFindRModelError(f"Unsupported MXNet device type {dev_type}.")
        if type_flag != 0:
            raise FinFindRModelError(
                f"Unsupported MXNet data type flag {type_flag}; FinFindR requires float32."
            )
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        data = np.frombuffer(_read_exact(stream, elements * 4), dtype="<f4").copy()
        arrays.append(data.reshape(shape))

    (name_count,) = _unpack(stream, "Q")
    if name_count != count:
        raise FinFindRModelError(
            f"MXNet parameter/name count mismatch: {count} arrays, {name_count} names."
        )
    names = []
    for _ in range(name_count):
        (length,) = _unpack(stream, "Q")
        if length > 1_000_000:
            raise FinFindRModelError("Unreasonable MXNet parameter name length.")
        names.append(_read_exact(stream, length).decode("utf-8"))

    if stream.read(1):
        raise FinFindRModelError("Unexpected trailing data in MXNet parameter file.")

    result: Dict[str, np.ndarray] = {}
    for raw_name, array in zip(names, arrays):
        name = raw_name.split(":", 1)[1] if raw_name.startswith(("arg:", "aux:")) else raw_name
        if name in result:
            raise FinFindRModelError(f"Duplicate MXNet parameter name: {name}.")
        result[name] = array
    return result

