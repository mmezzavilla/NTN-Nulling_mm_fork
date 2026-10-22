import numpy as np

from beacon.precoder import zc_precoder


def test_constant_modulus_unit_norm():
    w = zc_precoder()
    assert np.allclose(np.abs(w), 1 / 8) and np.isclose(np.linalg.norm(w), 1)


def test_flat_on_dft_grid():
    w = zc_precoder()
    F = np.kron(np.fft.fft(np.eye(4)), np.fft.fft(np.eye(16))) / 8  # unit-norm 2D DFT beams
    assert np.allclose(np.abs(F.conj().T @ w) ** 2, 1 / 64)


def test_shifted_set_shapes_and_modulus():
    from beacon.precoder import zc_precoder_set
    for k in (1, 2, 4):
        W = zc_precoder_set(k)
        assert W.shape == (k, 64) and np.allclose(np.abs(W), 1 / 8)
    assert np.allclose(zc_precoder_set(1)[0], zc_precoder())


def test_shifted_set_fills_half_bin_grid():
    # On the 2x-oversampled DFT grid, the 4-pattern average is much flatter than one pattern.
    from beacon.precoder import zc_precoder_set
    F = np.kron(np.fft.fft(np.eye(4), 8), np.fft.fft(np.eye(16), 32)).T / 8  # (256, 64) beams
    one = np.abs(F @ zc_precoder().conj()) ** 2
    avg = np.mean(np.abs(F @ zc_precoder_set(4).conj().T) ** 2, axis=1)
    assert np.isclose(avg.mean(), one.mean())                               # same total energy
    assert avg.min() > 4 * one.min()
