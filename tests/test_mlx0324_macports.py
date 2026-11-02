import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest


@pytest.mark.parametrize('codec,limit', [('q4_0', 0.3), ('q8_0', 0.025)])
def test_odd_rows_quantize_and_matmul(codec, limit):
    mx.random.seed(29)
    w = mx.random.normal((7, 64)).astype(mx.float32)
    x = mx.random.normal((3, 64)).astype(mx.float32)
    packed, scales = kq.quantize(w, codec)
    restored = kq.dequantize(packed, scales, codec).astype(mx.float32)
    actual = kq.quantized_matmul(x, packed, scales, codec, transpose=True).astype(mx.float32)
    expected = x @ restored.T
    mx.eval(w, restored, actual, expected)
    assert np.max(np.abs(np.array(w) - np.array(restored))) < limit
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=0.1, rtol=0.04)
