"""Layer A: link-level beacon simulation over a flat single-antenna channel.

The beacon has n = round(T·B) samples: a known unit-modulus preamble of
n_p = round(preamble_frac·n) samples, then a 128-bit nonce, LDPC coded
(5G LDPC, k=128, n=512, 20 BP iterations), QPSK-mapped to 256 symbols and repeated cyclically
over the remaining samples. gamma is the total received beacon energy over N0.

With n_seg > 1 the beacon is split into n_seg equal segments, one per cycled precoder.
Each segment has its own preamble and nonce repetitions and sees its own channel phase
(equal energy per segment). The detector sums the segments' preamble energies
non-coherently; the decoder sums the segments' LLRs.

The receiver's matched-filter output and the per-symbol sums over repetitions
are sufficient statistics, so they are drawn directly from their exact
distributions instead of generating every sample.
"""
import functools
import warnings

import numpy as np

K_NONCE = 128
N_CODE = 512
N_SYM = N_CODE // 2
BATCH = 5000


def detection_threshold(pfa, n_seg=1):
    """Threshold on Σ_seg |p^H r_p|² / (N0‖p‖²), which is Gamma(n_seg, 1) under noise only."""
    from scipy.special import gammainccinv
    return -np.log(pfa) if n_seg == 1 else float(gammainccinv(n_seg, pfa))


@functools.lru_cache(maxsize=1)
def _codec():
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
    enc = LDPC5GEncoder(k=K_NONCE, n=N_CODE)
    return enc, LDPC5GDecoder(enc, num_iter=20, hard_out=True)


def _sizes(t_s, b_hz, preamble_frac, n_seg=1):
    """Samples per segment n, preamble samples n_p, and repetitions per nonce symbol."""
    n = int(round(t_s * b_hz / n_seg))
    n_p = int(round(preamble_frac * n))
    if n - n_p < N_SYM:
        raise ValueError(f"beacon of {n} samples cannot carry one {N_SYM}-symbol nonce codeword")
    reps = np.full(N_SYM, (n - n_p) // N_SYM)
    reps[: (n - n_p) % N_SYM] += 1
    return n, n_p, reps


def _cn(rng, shape, var):
    return np.sqrt(var / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _run_batch(rng, gamma, n, n_p, reps, thr, m, n_seg=1):
    import tensorflow as tf
    enc, dec = _codec()
    amp = np.sqrt(gamma / (n * n_seg))              # per-sample amplitude, N0 = 1

    bits = rng.integers(0, 2, (m, K_NONCE)).astype(np.float32)
    c = enc(tf.constant(bits)).numpy()
    x = ((1 - 2 * c[:, 0::2]) + 1j * (1 - 2 * c[:, 1::2])) / np.sqrt(2)

    stat, llr0 = np.zeros(m), np.zeros((m, N_SYM), complex)
    for _ in range(n_seg):
        g = np.exp(1j * rng.uniform(0, 2 * np.pi, m))   # unknown channel phase of this segment
        z = amp * g * n_p + _cn(rng, m, n_p)            # p^H r_p
        stat += np.abs(z) ** 2 / n_p
        g_hat = z / n_p                                 # estimate of amp·g
        s = amp * g[:, None] * reps * x + _cn(rng, (m, N_SYM), reps)   # sums over repetitions
        u = s * np.conj(g_hat)[:, None] / np.abs(g_hat)[:, None]
        llr0 += 2 * np.sqrt(2) * np.abs(g_hat)[:, None] * u          # log p(0)/p(1) per I/Q bit
    detected = stat >= thr
    logits = np.empty((m, N_CODE), np.float32)
    logits[:, 0::2] = -llr0.real
    logits[:, 1::2] = -llr0.imag
    correct = np.all(dec(tf.constant(logits)).numpy() == bits, axis=1)
    return detected, correct


def simulate_beacon(gamma_db, t_s, b_hz, n_trials, pfa=1e-8, seed=0, preamble_frac=0.25,
                    n_seg=1):
    """P_det, nonce BLER given detection, and P(detected and nonce correct) vs gamma."""
    n, n_p, reps = _sizes(t_s, b_hz, preamble_frac, n_seg)
    thr = detection_threshold(pfa, n_seg)
    rng = np.random.default_rng(seed)
    gamma_db = np.asarray(gamma_db, float)
    pd, bler, psucc = (np.empty(gamma_db.shape) for _ in range(3))
    for i, gdb in enumerate(gamma_db.flat):
        det, ok = [], []
        for start in range(0, n_trials, BATCH):
            d, c = _run_batch(rng, 10 ** (gdb / 10), n, n_p, reps, thr,
                              min(BATCH, n_trials - start), n_seg)
            det.append(d)
            ok.append(c)
        det, ok = np.concatenate(det), np.concatenate(ok)
        pd.flat[i] = det.mean()
        bler.flat[i] = 1 - ok[det].mean() if det.any() else np.nan
        psucc.flat[i] = (det & ok).mean()
    return {"gamma_db": gamma_db, "pd": pd, "bler": bler, "psucc": psucc}


def measure_pfa(t_s, b_hz, pfa, n_trials, seed=0, preamble_frac=0.25, n_seg=1):
    """Fraction of noise-only trials that cross the detection threshold."""
    _, n_p, _ = _sizes(t_s, b_hz, preamble_frac, n_seg)
    z = _cn(np.random.default_rng(seed), (n_trials, n_seg), n_p)
    stat = np.sum(np.abs(z) ** 2, axis=1) / n_p
    return float(np.mean(stat >= detection_threshold(pfa, n_seg)))


def gamma_req_db(gamma_db, psucc, target=0.99):
    """gamma (dB) at the first crossing of target, linearly interpolated."""
    above = np.nonzero(np.asarray(psucc) >= target)[0]
    if above.size == 0:
        raise ValueError(f"success probability never reaches {target}")
    i = above[0]
    if i == 0:
        warnings.warn(f"success already ≥ {target} at the first grid point; "
                      f"gamma_req clamped to {gamma_db[0]} dB (extend the grid down)")
        return float(gamma_db[0])
    g0, g1, p0, p1 = gamma_db[i - 1], gamma_db[i], psucc[i - 1], psucc[i]
    return float(g0 + (target - p0) * (g1 - g0) / (p1 - p0))


def combined_gamma_req_db(gamma_db, psucc_list, target=0.99):
    """Conservative gamma_req over several beacon lengths: the max; raises if any is unreached."""
    return max(gamma_req_db(gamma_db, p, target) for p in psucc_list)
