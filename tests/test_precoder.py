import numpy as np

from beacon.precoder import zc_precoder


def test_constant_modulus_unit_norm():
    w = zc_precoder()
    assert np.allclose(np.abs(w), 1 / 8) and np.isclose(np.linalg.norm(w), 1)


def test_flat_on_dft_grid():
    w = zc_precoder()
    F = np.kron(np.fft.fft(np.eye(4)), np.fft.fft(np.eye(16))) / 8  # unit-norm 2D DFT beams
    assert np.allclose(np.abs(F.conj().T @ w) ** 2, 1 / 64)
