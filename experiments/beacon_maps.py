"""STEP 1 maps: which VSATs decode the gNB beacon, for P x T, plus sanity checks.

Run from the repo root:
  python -m experiments.beacon_maps --channels results/<..>_beacon_channels --curves results/<..>_beacon_curves
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beacon.io import make_run_dir, require_full_run, save_figure
from beacon.linkbudget import (GREEN, GREY, RED, beacon_snr, class_fractions, classify,
                               dbm_to_w, inr_worst, t_rule, t_rule_bound, t_rule_quantile)
from beacon.precoder import zc_precoder_set

B_HZ = 100e6
NF_VSAT_DB = 2.0
INR_TH_DB = -6.0
P_DBM = [30.0, 40.0, 45.0]
T_PAPER = 20e-6
Q_COVER = 0.99
COLORS = {GREEN: "#2e7d32", RED: "#c62828", GREY: "#9e9e9e"}
NAMES = {GREEN: "green", RED: "red", GREY: "grey"}


def ceil_10us(t):
    return float(np.ceil(t / 10e-6) * 10e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--band", choices=["wide", "narrow"], default=None,
                    help="wide: average over subcarriers (key hf); narrow: carrier only (key h). "
                         "Default: wide if available.")
    ap.add_argument("--n-precoders", type=int, default=1,
                    help="number of half-bin-shifted ZC precoders cycled over the beacon (1-4)")
    ap.add_argument("--allow-smoke", action="store_true")
    args = ap.parse_args()
    for d in (args.channels, args.curves):
        require_full_run(d, args.allow_smoke)

    ch = np.load(pathlib.Path(args.channels) / "channels.npz")
    band = args.band or ("wide" if "hf" in ch else "narrow")
    h_all = ch["hf" if band == "wide" else "h"]       # (n_drops, n_vsat, 3, 64[, K])
    gamma_req_db = float(np.load(pathlib.Path(args.curves) / "curves.npz")["gamma_req_db"])
    gamma_req, inr_th = 10 ** (gamma_req_db / 10), 10 ** (INR_TH_DB / 10)
    w = zc_precoder_set(args.n_precoders)
    n_sect, n_tx = h_all.shape[2], h_all.shape[3]

    rates = {p: [beacon_snr(h, w, dbm_to_w(p), 1.0, NF_VSAT_DB) for h in h_all] for p in P_DBM}
    inrs = {p: [inr_worst(h, dbm_to_w(p), B_HZ, NF_VSAT_DB) for h in h_all] for p in P_DBM}
    pooled = {p: (np.concatenate(rates[p]), np.concatenate(inrs[p])) for p in P_DBM}
    t_all_raw = max(t_rule(r, i, gamma_req, inr_th) for r, i in pooled.values())
    t_q_raw = max(t_rule_quantile(r, i, gamma_req, inr_th, Q_COVER) for r, i in pooled.values())
    bound = t_rule_bound(gamma_req, n_tx, n_sect, inr_th, B_HZ)
    t_values = [T_PAPER, ceil_10us(bound), ceil_10us(t_q_raw), ceil_10us(t_all_raw)]
    t_labels = ["paper", "analytic bound", f"{Q_COVER:.0%} of harmed", "all harmed"]

    classes = np.array([[[classify(r * t, i, gamma_req, inr_th)
                          for r, i in zip(rates[p], inrs[p])] for p in P_DBM] for t in t_values])
    frac = class_fractions(classes, axis=(2, 3))      # (n_T, n_P, 3)

    params = dict(channels=str(args.channels), curves=str(args.curves), band=band, b_hz=B_HZ,
                  nf_vsat_db=NF_VSAT_DB, nf_vsat_notebook_db=3.0, inr_th_db=INR_TH_DB,
                  p_dbm=P_DBM, t_values_s=t_values, t_labels=t_labels,
                  gamma_req_db=gamma_req_db, t_all_harmed_raw_s=t_all_raw,
                  t_quantile=Q_COVER, t_quantile_raw_s=t_q_raw, t_rule_bound_s=bound,
                  precoder="2D Zadoff-Chu 4x16, root 1", n_precoders=args.n_precoders,
                  precoder_shifts="half-bin (rows, cols): (0,0), (0,.5), (.5,0), (.5,.5)",
                  allow_smoke=args.allow_smoke)
    suffix = f"_{args.n_precoders}prec" if args.n_precoders > 1 else ""
    run_dir = make_run_dir(f"beacon_maps_{band}band{suffix}", params)
    np.savez(run_dir / "classes.npz", classes=classes, t_values=np.array(t_values),
             p_dbm=np.array(P_DBM), fractions=frac, t_rule_bound=bound,
             t_quantile=t_q_raw, t_all_harmed=t_all_raw)

    plot_maps(run_dir, ch, classes, t_values, t_labels)
    plot_fractions(run_dir, frac, t_values, t_labels)

    print(f"band = {band}, gamma_req = {gamma_req_db:.2f} dB")
    print(f"T: analytic bound {bound * 1e6:.1f} us, {Q_COVER:.0%} of harmed {t_q_raw * 1e6:.1f} us, "
          f"all harmed {t_all_raw * 1e6:.1f} us")
    for ti, t in enumerate(t_values):
        for pi, p in enumerate(P_DBM):
            f = frac[ti, pi]
            print(f"T={t * 1e6:.0f} us ({t_labels[ti]}) P={p:.0f} dBm: green {f[GREEN]:.3f} "
                  f"red {f[RED]:.3f} grey {f[GREY]:.3f}")
    ok = sanity_checks(frac, classes, t_all_raw, bound)
    print(f"saved {run_dir}")
    sys.exit(0 if ok else 1)


def plot_maps(run_dir, ch, classes, t_values, t_labels):
    fig, axes = plt.subplots(len(P_DBM), len(t_values), figsize=(4.2 * len(t_values), 13),
                             sharex=True, sharey=True, squeeze=False)
    pos, bs = ch["vsat_pos"][0], ch["bs_pos"][0]
    for pi, p in enumerate(P_DBM):
        for ti, t in enumerate(t_values):
            ax, c = axes[pi, ti], classes[ti, pi, 0]
            ax.imshow(ch["point_type"], extent=ch["extent"], cmap="Greys", vmin=0, vmax=6,
                      origin="upper", alpha=0.6)
            for k in (GREY, GREEN, RED):
                m = c == k
                ax.scatter(pos[m, 0], pos[m, 1], s=10, color=COLORS[k],
                           label=f"{NAMES[k]} ({m.sum()})")
            ax.plot(bs[0], bs[1], "k^", ms=8)
            ax.set_title(f"P = {p:.0f} dBm, T = {t * 1e6:.0f} µs\n({t_labels[ti]})", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    fig.tight_layout()
    save_figure(fig, run_dir, "maps_grid")


def plot_fractions(run_dir, frac, t_values, t_labels):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ti, (t, ls) in enumerate(zip(t_values, ["-", "--", "-.", ":"])):
        for k in (GREEN, RED):
            ax.plot(P_DBM, frac[ti, :, k], ls, marker="o", color=COLORS[k],
                    label=f"{NAMES[k]}, T = {t * 1e6:.0f} µs ({t_labels[ti]})")
    ax.set_xlabel("gNB TX power per sector (dBm)")
    ax.set_ylabel("Fraction of VSATs")
    ax.set_xticks(P_DBM)
    ax.grid(True)
    ax.legend(fontsize=7)
    save_figure(fig, run_dir, "fractions")


def sanity_checks(frac, classes, t_all_raw, bound):
    ok = True
    for ti in range(frac.shape[0]):
        mono = bool(np.all(np.diff(frac[ti, :, GREEN]) >= 0))
        ok &= mono
        print(f"{'PASS' if mono else 'FAIL'} (a) green fraction non-decreasing in P, T index {ti}")
    # Spec check 4: red at the analytic length tests the ‖h‖²/N_tx assumption. Reported, not failed.
    n_red, n = int(np.sum(classes[1] == RED)), classes[1].size
    print(f"NOTE (b) red at analytic bound: {n_red} of {n} VSAT-power cases ({n_red / n:.4f})")
    print(f"NOTE (c) all-harmed T / analytic bound = {t_all_raw / bound:.2f}")
    return ok


if __name__ == "__main__":
    main()
