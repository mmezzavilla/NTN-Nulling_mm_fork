import numpy as np

from beacon.channels import channels_from_cir


def test_sums_paths():
    a = np.zeros((2, 1, 3, 64, 5, 1), complex)
    a[1, 0, 2, 7, :, 0] = 0.1
    h = channels_from_cir(a)
    assert h.shape == (2, 3, 64) and np.isclose(h[1, 2, 7], 0.5)


def test_single_rx_keeps_shape():
    assert channels_from_cir(np.ones((1, 1, 3, 64, 1, 1))).shape == (1, 3, 64)


def test_channels_from_cfr_keeps_subcarriers():
    from beacon.channels import channels_from_cfr
    cfr = np.zeros((2, 1, 3, 64, 1, 8), complex)
    cfr[1, 0, 2, 7, 0, 5] = 0.3
    hf = channels_from_cfr(cfr)
    assert hf.shape == (2, 3, 64, 8) and np.isclose(hf[1, 2, 7, 5], 0.3) and hf.dtype == np.complex64
