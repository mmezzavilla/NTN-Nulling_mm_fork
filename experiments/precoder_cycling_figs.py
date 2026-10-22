"""Compare beacon coverage when cycling K half-bin-shifted ZC precoders (K = 1, 2, 4).

Each K uses its own gamma_req from the K-segment Layer A run.
Run from the repo root:
  python -m experiments.precoder_cycling_figs --channels results/<..>_beacon_channels \
      --curves 1=results/2026-10-14_beacon_curves 2=results/<..>_beacon_curves_2seg \
               4=results/<..>_beacon_curves_4seg
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beacon.io import make_run_dir, require_full_run, save_figure
from beacon.linkbudget import beacon_snr, dbm_to_w, inr_worst, t_rule_bound
from beacon.precoder import zc_precoder_set

B_HZ, NF_DB, INR_TH_DB = 100e6, 2.0, -6.0
P_DBM = [30.0, 40.0, 45.0]
K_COL = {1: "#8a9097", 2: "#d4831a", 4: "#1d5f8a"}
INK, MUTED = "#22262b", "#6b7179"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e3e5e8", "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--curves", nargs="+", required=True, help="K=run_dir pairs")
    ap.add_argument("--allow-smoke", action="store_true")
    args = ap.parse_args()
    curves = {int(k): v for k, v in (c.split("=", 1) for c in args.curves)}
    for d in [args.channels, *curves.values()]:
        require_full_run(d, args.allow_smoke)

    h_all = np.load(pathlib.Path(args.channels) / "channels.npz")["hf"]
    n_sect, n_tx = h_all.shape[2], h_all.shape[3]
    inr_th = 10 ** (INR_TH_DB / 10)
    gq = {k: float(np.load(pathlib.Path(d) / "curves.npz")["gamma_req_db"]) for k, d in curves.items()}
    run_dir = make_run_dir("precoder_cycling", dict(channels=args.channels, curves=curves,
                                                    gamma_req_db=gq, b_hz=B_HZ, nf_db=NF_DB,
                                                    inr_th_db=INR_TH_DB, p_dbm=P_DBM))

    # Total sector gain ‖h‖² summed over sectors, per VSAT (wideband mean), pooled over drops.
    tot = (np.abs(h_all) ** 2).sum(axis=3).mean(axis=-1).sum(axis=-1).reshape(-1) / n_tx
    stats, t_req, rho = {}, {}, {}
    for k in curves:
        w = zc_precoder_set(k)
        gamma_req = 10 ** (gq[k] / 10)
        best1 = np.concatenate([beacon_snr(h, w, 1.0, 1.0, 0.0) for h in h_all]).max(axis=1)
        bound = t_rule_bound(gamma_req, n_tx, n_sect, inr_th, B_HZ)
        stats[k] = dict(gamma_req_db=gq[k], t_bound_s=bound)
        for p in P_DBM:
            rate = np.concatenate([beacon_snr(h, w, dbm_to_w(p), 1.0, NF_DB) for h in h_all])
            inr = np.concatenate([inr_worst(h, dbm_to_w(p), B_HZ, NF_DB) for h in h_all])
            harmed = inr >= inr_th
            t = gamma_req / rate.max(axis=1)[harmed]
            t_req[k, p] = t
            stats[k][f"{p:.0f}"] = dict(
                n_harmed=int(harmed.sum()), t99_s=float(np.quantile(t, 0.99)),
                t_all_s=float(t.max()), red_frac_at_bound=float(np.mean(t > bound) * harmed.mean()),
                red_frac_at_20us=float(np.mean(t > 20e-6) * harmed.mean()))
            if p == P_DBM[0]:
                # beacon_snr at NF 0 dB divides by kT; undo it to compare with the raw gain.
                rho[k] = 10 * np.log10(best1[harmed] * dbm_to_w(-174.0) / tot[harmed])
        print(f"K={k}: gamma_req {gq[k]:.2f} dB, bound {bound * 1e3:.2f} ms, " + ", ".join(
            f"{p:.0f} dBm: 99% {stats[k][f'{p:.0f}']['t99_s'] * 1e3:.2f} ms / all "
            f"{stats[k][f'{p:.0f}']['t_all_s'] * 1e3:.1f} ms / red@bound "
            f"{stats[k][f'{p:.0f}']['red_frac_at_bound']:.4f}" for p in P_DBM))
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, p in zip(axes, P_DBM):
        for k in sorted(curves):
            t = np.sort(t_req[k, p])
            ax.step(t * 1e3, 1 - np.arange(t.size) / t.size, where="post", color=K_COL[k],
                    label=f"K = {k}" + (" (today)" if k == 1 else ""))
            ax.axvline(stats[k]["t_bound_s"] * 1e3, color=K_COL[k], lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.2)
        ax.set_title(f"P = {p:.0f} dBm per sector", fontsize=9.5)
        ax.set_xlabel("Beacon length T needed to decode (ms)")
    axes[0].set_ylabel("Share of harmed VSATs still missed")
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save_figure(fig, run_dir, "cycling_treq_ccdf")

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for k in sorted(curves):
        r = np.sort(rho[k])
        ax.plot(r, np.arange(1, r.size + 1) / r.size, color=K_COL[k], label=f"K = {k}")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Best-sector beacon gain relative to Σ_s ‖h_s‖² / N_tx (dB)")
    ax.set_ylabel("CDF over harmed VSATs")
    ax.set_xlim(-30, 10)
    ax.legend(fontsize=8.5, loc="upper left")
    save_figure(fig, run_dir, "cycling_ripple")
    for k in sorted(curves):
        print(f"ripple K={k}: median {np.median(rho[k]):+.1f} dB, 1% {np.quantile(rho[k], .01):+.1f} dB, "
              f"min {rho[k].min():+.1f} dB")
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
