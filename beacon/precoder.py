"""Quasi-omni beacon precoder: constant modulus, flat on the 2D DFT beam grid."""
import numpy as np


def _zc(n, root):
    k = np.arange(n)
    return np.exp(-1j * np.pi * root * k * (k + n % 2) / n)


def zc_precoder(n_rows=4, n_cols=16, root=1):
    """2D Zadoff-Chu precoder (row-major Kronecker), unit norm, all PAs at equal power."""
    return np.kron(_zc(n_rows, root), _zc(n_cols, root)) / np.sqrt(n_rows * n_cols)
