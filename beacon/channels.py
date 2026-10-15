"""gNB -> VSAT channel vectors from Sionna RT paths: narrowband (CIR) or per subcarrier (CFR)."""
import numpy as np


def channels_from_cir(a):
    """Sum paths of a [n_rx, 1, n_tx, n_tx_ant, n_paths, n_time] CIR at time 0.

    Returns (n_rx, n_tx, n_tx_ant) complex64. No squeeze, so a single receiver
    keeps its axis.
    """
    return np.asarray(a)[:, 0, :, :, :, 0].sum(axis=-1).astype(np.complex64)


def channels_from_cfr(cfr):
    """Per-subcarrier channels from a [n_rx, 1, n_tx, n_tx_ant, n_time, n_sc] CFR at time 0.

    Returns (n_rx, n_tx, n_tx_ant, n_sc) complex64.
    """
    return np.asarray(cfr)[:, 0, :, :, 0, :].astype(np.complex64)
