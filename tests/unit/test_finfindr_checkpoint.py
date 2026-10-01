import struct

import numpy as np
import pytest

from core.vision.finfindr.checkpoint import read_mxnet_params
from core.vision.finfindr.errors import FinFindRModelError


def _checkpoint_bytes(name="arg:weight", values=None):
    values = np.asarray(values if values is not None else [[1.5, -2.0]], dtype="<f4")
    encoded = name.encode("utf-8")
    return b"".join(
        (
            struct.pack("<QQQ", 0x112, 0, 1),
            struct.pack("<IiI", 0xF993FAC9, 0, values.ndim),
            struct.pack("<" + "q" * values.ndim, *values.shape),
            struct.pack("<iii", 1, 0, 0),
            values.tobytes(),
            struct.pack("<QQ", 1, len(encoded)),
            encoded,
        )
    )


def test_reads_dense_float32_checkpoint_and_strips_argument_prefix(tmp_path):
    path = tmp_path / "test.params"
    path.write_bytes(_checkpoint_bytes())
    result = read_mxnet_params(path)
    assert set(result) == {"weight"}
    np.testing.assert_array_equal(result["weight"], np.array([[1.5, -2.0]], dtype=np.float32))


def test_rejects_trailing_or_truncated_data(tmp_path):
    path = tmp_path / "bad.params"
    path.write_bytes(_checkpoint_bytes() + b"x")
    with pytest.raises(FinFindRModelError, match="trailing data"):
        read_mxnet_params(path)
    path.write_bytes(_checkpoint_bytes()[:-3])
    with pytest.raises(FinFindRModelError, match="Unexpected end"):
        read_mxnet_params(path)

