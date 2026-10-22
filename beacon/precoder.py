"""Quasi-omni beacon precoder: constant modulus, flat on the 2D DFT beam grid."""
import numpy as np


def _zc(n, root):
    k = np.arange(n)
    return np.exp(-1j * np.pi * root * k * (k + n % 2) / n)


def zc_precoder(n_rows=4, n_cols=16, root=1):
    """2D Zadoff-Chu precoder (row-major Kronecker), unit norm, all PAs at equal power."""
    return np.kron(_zc(n_rows, root), _zc(n_cols, root)) / np.sqrt(n_rows * n_cols)


# Half-bin shifts (rows, cols) of the 2D DFT grid; the first entry is the unshifted pattern.
_SHIFTS = [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]


def zc_precoder_set(k, n_rows=4, n_cols=16, root=1):
    """k ZC precoders, each steered by a half-bin shift so their nulls fall on different directions.

    Cycling them across the beacon's repetitions (equal time each) averages the pattern
    between DFT grid points. Shape (k, n_rows*n_cols), each row unit norm.
    """
    if not 1 <= k <= len(_SHIFTS):
        raise ValueError(f"k must be 1..{len(_SHIFTS)}")
    w = zc_precoder(n_rows, n_cols, root)
    r, c = np.arange(n_rows), np.arange(n_cols)
    ramps = [np.kron(np.exp(2j * np.pi * sr * r / n_rows), np.exp(2j * np.pi * sc * c / n_cols))
             for sr, sc in _SHIFTS[:k]]
    return np.stack([w * ramp for ramp in ramps])
