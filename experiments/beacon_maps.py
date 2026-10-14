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

from beacon.io import make_run_dir, save_figure
from beacon.linkbudget import (GREEN, GREY, RED, beacon_snr, classify, dbm_to_w,
                               inr_worst, t_rule, t_rule_bound)
from beacon.precoder import zc_precoder

B_HZ = 100e6
NF_VSAT_DB = 2.0
INR_TH_DB = -6.0
P_DBM = [30.0, 40.0, 45.0]
T_PAPER = 20e-6
COLORS = {GREEN: "#2e7d32", RED: "#c62828", GREY: "#9e9e9e"}
NAMES = {GREEN: "green", RED: "red", GREY: "grey"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--curves", required=True)
    args = ap.parse_args()

    ch = np.load(pathlib.Path(args.channels) / "channels.npz")
    h_all = ch["h"]                                   # (n_drops, n_vsat, 3, 64)
    gamma_req_db = float(np.load(pathlib.Path(args.curves) / "curves.npz")["gamma_req_db"])
    gamma_req, inr_th = 10 ** (gamma_req_db / 10), 10 ** (INR_TH_DB / 10)
    w = zc_precoder()
    n_drops, _, n_sect, n_tx = h_all.shape

    rates = {p: [beacon_snr(h, w, dbm_to_w(p), 1.0, NF_VSAT_DB) for h in h_all] for p in P_DBM}
    inrs = {p: [inr_worst(h, dbm_to_w(p), B_HZ, NF_VSAT_DB) for h in h_all] for p in P_DBM}
    t_rule_raw = max(t_rule(r, i, gamma_req, inr_th)
                     for p in P_DBM for r, i in zip(rates[p], inrs[p]))
    t_rule_s = np.ceil(t_rule_raw / 10e-6) * 10e-6
    bound = t_rule_bound(gamma_req, n_tx, n_sect, inr_th, B_HZ)
    t_values = [T_PAPER, t_rule_s]

    classes = np.array([[[classify(r * t, i, gamma_req, inr_th)
                          for r, i in zip(rates[p], inrs[p])] for p in P_DBM] for t in t_values])

    params = dict(channels=str(args.channels), curves=str(args.curves), b_hz=B_HZ,
                  nf_vsat_db=NF_VSAT_DB, nf_vsat_notebook_db=3.0, inr_th_db=INR_TH_DB,
                  p_dbm=P_DBM, t_values_s=t_values, gamma_req_db=gamma_req_db,
                  t_rule_s=t_rule_s, t_rule_raw_s=t_rule_raw, t_rule_bound_s=bound,
                  precoder="2D Zadoff-Chu 4x16, root 1")
    run_dir = make_run_dir("beacon_maps", params)
    np.savez(run_dir / "classes.npz", classes=classes, t_values=np.array(t_values),
             p_dbm=np.array(P_DBM), t_rule=t_rule_s, t_rule_bound=bound)

    plot_maps(run_dir, ch, classes, t_values)
    frac = plot_fractions(run_dir, classes, t_values)

    print(f"gamma_req = {gamma_req_db:.2f} dB, T_rule = {t_rule_s * 1e6:.0f} us "
          f"(raw {t_rule_raw * 1e6:.1f} us), analytic bound = {bound * 1e6:.1f} us")
    for ti, t in enumerate(t_values):
        for pi, p in enumerate(P_DBM):
            f = frac[ti, pi]
            print(f"T={t * 1e6:.0f} us P={p:.0f} dBm: green {f[GREEN]:.3f} red {f[RED]:.3f} "
                  f"grey {f[GREY]:.3f}")
    ok = sanity_checks(frac, classes, t_rule_raw, bound)
    print(f"saved {run_dir}")
    sys.exit(0 if ok else 1)


def plot_maps(run_dir, ch, classes, t_values):
    fig, axes = plt.subplots(len(P_DBM), len(t_values), figsize=(9, 13), sharex=True, sharey=True)
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
            ax.set_title(f"P = {p:.0f} dBm, T = {t * 1e6:.0f} µs", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    fig.tight_layout()
    save_figure(fig, run_dir, "maps_grid")


def plot_fractions(run_dir, classes, t_values):
    frac = np.stack([(classes == k).mean(axis=(2, 3)) for k in (GREY, GREEN, RED)], axis=-1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ti, (t, ls) in enumerate(zip(t_values, ["-", "--"])):
        for k in (GREEN, RED, GREY):
            ax.plot(P_DBM, frac[ti, :, k], ls, marker="o", color=COLORS[k],
                    label=f"{NAMES[k]}, T = {t * 1e6:.0f} µs")
    ax.set_xlabel("gNB TX power per sector (dBm)")
    ax.set_ylabel("Fraction of VSATs")
    ax.set_xticks(P_DBM)
    ax.grid(True)
    ax.legend(fontsize=8)
    save_figure(fig, run_dir, "fractions")
    return frac


def sanity_checks(frac, classes, t_rule_raw, bound):
    ok = True
    for ti in range(frac.shape[0]):
        mono = bool(np.all(np.diff(frac[ti, :, GREEN]) >= 0))
        ok &= mono
        print(f"{'PASS' if mono else 'FAIL'} (a) green fraction non-decreasing in P, T index {ti}")
    n_red = int(np.sum(classes[1] == RED))
    ok &= n_red == 0
    print(f"{'PASS' if n_red == 0 else 'FAIL'} (b) red count at T_rule = {n_red}")
    print(f"NOTE (c) T_rule / analytic bound = {t_rule_raw / bound:.2f}"
          + ("  (>1: precoder ripple exceeds the ‖h‖²/N_tx assumption)" if t_rule_raw > bound else ""))
    return ok


if __name__ == "__main__":
    main()
