"""Beacon link budget and green/red/grey VSAT classification (pure numpy).

Channels h have shape (n_vsat, n_sect, n_tx); the precoder w has shape (n_tx,).
Powers are in W, times in s, bandwidths in Hz; SNR/INR values are linear.
"""
import numpy as np

GREY, GREEN, RED = 0, 1, 2


def dbm_to_w(p_dbm):
    return 10 ** ((p_dbm - 30) / 10)


def noise_psd_w(nf_db, kt_dbm_hz=-174.0):
    """Receiver noise PSD in W/Hz."""
    return dbm_to_w(kt_dbm_hz + nf_db)


def inr_worst(h, p_w, b_hz, nf_db):
    """Worst-case data INR: every sector's full beam on the VSAT, summed over sectors."""
    gain = np.sum(np.abs(h) ** 2, axis=(1, 2))
    return p_w * gain / (noise_psd_w(nf_db) * b_hz)


def beacon_snr(h, w, p_w, t_s, nf_db):
    """Total received beacon energy over N0, per VSAT and sector."""
    return p_w * t_s * np.abs(h @ w.conj()) ** 2 / noise_psd_w(nf_db)


def classify(gamma, inr, gamma_req, inr_th):
    """GREEN if any sector's beacon decodes, else RED if harmed, else GREY."""
    green = np.any(gamma >= gamma_req, axis=1)
    harmed = inr >= inr_th
    return np.where(green, GREEN, np.where(harmed, RED, GREY)).astype(np.int8)


def t_rule(snr_rate, inr, gamma_req, inr_th):
    """Shortest beacon length (s) at which every harmed VSAT decodes some sector.

    snr_rate is beacon_snr evaluated at t_s = 1 s.
    """
    harmed = inr >= inr_th
    if not np.any(harmed):
        return 0.0
    best = np.max(snr_rate[harmed], axis=1)
    with np.errstate(divide="ignore"):
        return float(np.max(gamma_req / best))


def t_rule_bound(gamma_req, n_tx, n_sect, inr_th, b_hz):
    """Analytic bound from the spec, assuming |w^H h_s|² = ‖h_s‖²/n_tx."""
    return n_sect * gamma_req * n_tx / (inr_th * b_hz)
