"""Narrowband gNB -> VSAT channel vectors from Sionna RT path coefficients."""
import numpy as np


def channels_from_cir(a):
    """Sum paths of a [n_rx, 1, n_tx, n_tx_ant, n_paths, n_time] CIR at time 0.

    Returns (n_rx, n_tx, n_tx_ant) complex64. No squeeze, so a single receiver
    keeps its axis.
    """
    return np.asarray(a)[:, 0, :, :, :, 0].sum(axis=-1).astype(np.complex64)
