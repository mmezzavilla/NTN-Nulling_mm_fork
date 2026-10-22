"""Beacon link budget and green/red/grey VSAT classification (pure numpy).

Channels h have shape (n_vsat, n_sect, n_tx) (narrowband, at the carrier) or
(n_vsat, n_sect, n_tx, n_sc) (wideband, per subcarrier); gains are averaged over
subcarriers. The precoder w has shape (n_tx,), or (k, n_tx) for a cycled set.
Powers are in W, times in s, bandwidths in Hz; SNR/INR values are linear.
"""
import numpy as np

GREY, GREEN, RED = 0, 1, 2


def dbm_to_w(p_dbm):
    return 10 ** ((p_dbm - 30) / 10)


def noise_psd_w(nf_db, kt_dbm_hz=-174.0):
    """Receiver noise PSD in W/Hz."""
    return dbm_to_w(kt_dbm_hz + nf_db)


def _wideband(h):
    return h[..., None] if h.ndim == 3 else h


def inr_worst(h, p_w, b_hz, nf_db):
    """Worst-case data INR: every sector's full beam on the VSAT, summed over sectors."""
    gain = np.mean(np.sum(np.abs(_wideband(h)) ** 2, axis=(1, 2)), axis=-1)
    return p_w * gain / (noise_psd_w(nf_db) * b_hz)


def beacon_snr(h, w, p_w, t_s, nf_db):
    """Total received beacon energy over N0, per VSAT and sector.

    w is one precoder (n_tx,) or a set (k, n_tx) cycled with equal time each, so the
    gain is averaged over the set.
    """
    w = np.atleast_2d(w)
    beam = np.einsum("nstk,pt->npsk", _wideband(h), w.conj())
    return p_w * t_s * np.mean(np.abs(beam) ** 2, axis=(1, -1)) / noise_psd_w(nf_db)


def classify(gamma, inr, gamma_req, inr_th):
    """GREEN if any sector's beacon decodes, else RED if harmed, else GREY."""
    green = np.any(gamma >= gamma_req, axis=1)
    harmed = inr >= inr_th
    return np.where(green, GREEN, np.where(harmed, RED, GREY)).astype(np.int8)


def t_required(snr_rate, inr, gamma_req, inr_th):
    """Beacon length (s) each harmed VSAT needs to decode its best sector.

    snr_rate is beacon_snr evaluated at t_s = 1 s. Raises if a harmed VSAT
    receives no beacon energy at all, since no beacon length can then reach it.
    """
    best = np.max(snr_rate[inr >= inr_th], axis=1)
    if np.any(best <= 0):
        raise ValueError(f"{np.sum(best <= 0)} harmed VSAT(s) receive zero beacon gain")
    return gamma_req / best


def t_rule(snr_rate, inr, gamma_req, inr_th):
    """Shortest beacon length (s) at which every harmed VSAT decodes some sector."""
    t = t_required(snr_rate, inr, gamma_req, inr_th)
    return float(t.max()) if t.size else 0.0


def t_rule_quantile(snr_rate, inr, gamma_req, inr_th, q):
    """Beacon length (s) that covers a fraction q of the harmed VSATs."""
    t = t_required(snr_rate, inr, gamma_req, inr_th)
    return float(np.quantile(t, q)) if t.size else 0.0


def t_rule_bound(gamma_req, n_tx, n_sect, inr_th, b_hz):
    """Analytic bound from the spec, assuming |w^H h_s|² = ‖h_s‖²/n_tx."""
    return n_sect * gamma_req * n_tx / (inr_th * b_hz)


def class_fractions(classes, axis):
    """Fraction of each class along axis; output's last axis is indexed by GREY/GREEN/RED."""
    out = np.zeros(np.delete(classes.shape, axis).tolist() + [3])
    for k in (GREY, GREEN, RED):
        out[..., k] = np.mean(classes == k, axis=axis)
    return out
