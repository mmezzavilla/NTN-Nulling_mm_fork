"""STEP 1 report figures from saved runs (no ray tracing, no link-level simulation).

Run from the repo root:
  python -m experiments.step1_report_figs --channels results/2026-10-15_beacon_channels \
      --channels-narrow results/2026-10-15_beacon_channels \
      --curves results/2026-10-14_beacon_curves --maps results/2026-10-15_beacon_maps_wideband
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beacon.io import make_run_dir, save_figure
from beacon.linkbudget import GREEN, GREY, RED, beacon_snr, dbm_to_w, inr_worst
from beacon.phy import detection_threshold
from beacon.precoder import zc_precoder

B_HZ, NF_DB, INR_TH_DB = 100e6, 2.0, -6.0
P_DBM = [30.0, 40.0, 45.0]
C = {GREEN: "#2e7d32", RED: "#c62828", GREY: "#b0b4b8"}
P_COL = {30.0: "#1f6fb2", 40.0: "#d4831a", 45.0: "#7a3fa0"}
INK, MUTED = "#22262b", "#6b7179"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e3e5e8", "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def per_vsat(h_all, w, gamma_req, inr_th):
    """Pooled per-VSAT quantities over drops: best-sector rate, INR and ripple ratio."""
    out = {}
    n_tx = h_all.shape[3]
    for p in P_DBM:
        rate = np.concatenate([beacon_snr(h, w, dbm_to_w(p), 1.0, NF_DB) for h in h_all])
        inr = np.concatenate([inr_worst(h, dbm_to_w(p), B_HZ, NF_DB) for h in h_all])
        harmed = inr >= inr_th
        best = rate.max(axis=1)
        out[p] = dict(t_req=gamma_req / best[harmed], harmed=harmed, inr=inr, best=best)
    # Ripple: best-sector quasi-omni gain vs the ‖h‖²/N_tx assumption summed over sectors.
    hw = h_all if h_all.ndim == 5 else h_all[..., None]
    tot = (np.abs(hw) ** 2).sum(axis=3).mean(axis=-1).sum(axis=-1).reshape(-1) / n_tx
    beam = np.abs(np.einsum("dvstk,t->dvsk", hw, w.conj())) ** 2
    rate1 = beam.mean(axis=-1).max(axis=-1).reshape(-1)
    return out, rate1, tot


def fig_curves(run_dir, cv, gamma_req_db):
    g = cv["gamma_db"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(g, cv["pd_t20us"], color="#1f6fb2", label="P_det (detected)")
    ax.plot(g, cv["psucc_t20us"], color=C[GREEN], label="P_succ (detected + nonce)")
    ax.plot(g, cv["psucc_t1ms"], color=INK, ls=(0, (1, 2)), lw=1.6,
            label="P_succ, 1 ms beacon")
    # The detector sees only the preamble (a quarter of the energy), so its threshold maps to
    # ln(1/P_FA) / 0.25 in total-beacon gamma.
    thr = 10 * np.log10(detection_threshold(1e-8) / 0.25)
    ax.axvline(thr, color=MUTED, lw=1, ls="--")
    ax.text(thr - 0.2, 0.78, f"detection threshold\nln(10⁸) on preamble\n= {thr:.1f} dB of total γ",
            ha="right", color=MUTED, fontsize=8.5)
    ax.axvline(gamma_req_db, color=C[RED], lw=1, ls="--")
    ax.text(gamma_req_db + 0.25, 0.55, f"γ_req = {gamma_req_db:.1f} dB\n(P_succ = 0.99)",
            color=C[RED], fontsize=8.5)
    ax.set_xlabel("Total received beacon energy over N0, γ = Es,tot/N0 (dB)")
    ax.set_ylabel("Probability")
    ax.set_xlim(14, 30)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.3), fontsize=8.5)
    save_figure(fig, run_dir, "fig1_linklevel")


def fig_maps(run_dir, ch, classes, t_values):
    cols = [0, 1, 2]
    names = ["20 µs (paper preamble)", "2.07 ms (analytic bound)", "10.7 ms (covers 99 % of harmed)"]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.6), sharex=True, sharey=True)
    pos, bs = ch["vsat_pos"][0], ch["bs_pos"][0]
    for pi, p in enumerate(P_DBM):
        for j, ti in enumerate(cols):
            ax, c = axes[pi, j], classes[ti, pi, 0]
            ax.imshow(ch["point_type"], extent=ch["extent"], cmap="Greys", vmin=0, vmax=8,
                      origin="upper", alpha=0.45)
            ax.grid(False)
            for k, lab in ((GREY, "harmless, missed"), (GREEN, "decoded"), (RED, "harmed, missed")):
                m = c == k
                ax.scatter(pos[m, 0], pos[m, 1], s=13 if k != GREY else 8, color=C[k],
                           edgecolors="white", linewidths=0.4, label=f"{lab} ({m.sum()})",
                           zorder=3 if k == RED else 2)
            ax.plot(bs[0], bs[1], marker="^", color=INK, ms=9, mec="white", zorder=4)
            ax.legend(loc="upper right", fontsize=7.2, handletextpad=0.2, borderpad=0.2,
                      facecolor="white", framealpha=0.85, frameon=True, edgecolor="none")
            if pi == 0:
                ax.set_title(f"T = {names[j]}", fontsize=9.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"P = {p:.0f} dBm\n\ny (m)", fontsize=9.5)
            if pi == 2:
                ax.set_xlabel("x (m)")
            ax.set_aspect("equal")
    fig.tight_layout()
    save_figure(fig, run_dir, "fig2_maps")


def fig_fractions(run_dir, frac, t_values):
    labels = ["20 µs\npaper", "2.07 ms\nbound", "10.7 ms\n99 %", f"{t_values[3] * 1e3:.0f} ms\nall"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    x = np.arange(len(t_values))
    for pi, (ax, p) in enumerate(zip(axes, P_DBM)):
        bottom = np.zeros(len(t_values))
        for k, lab in ((GREEN, "decoded"), (RED, "harmed, missed"), (GREY, "harmless, missed")):
            v = frac[:, pi, k] * 100
            ax.bar(x, v, 0.68, bottom=bottom, color=C[k], edgecolor="white", linewidth=1.5,
                   label=lab)
            for xi, (vi, bi) in enumerate(zip(v, bottom)):
                if k == RED and vi > 0.05:
                    ax.text(xi, bi + vi + 1.5, f"{vi:.1f}%", ha="center", va="bottom",
                            fontsize=8, color=C[RED], fontweight="bold")
            bottom += v
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"P = {p:.0f} dBm per sector", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of VSATs (%)")
    axes[0].set_ylim(0, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8.5)
    save_figure(fig, run_dir, "fig3_fractions")


def fig_treq(run_dir, wide, narrow, t_values):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for p in P_DBM:
        for d, ls, lab in ((wide, "-", ""), (narrow, (0, (3, 2)), " narrowband")):
            t = np.sort(d[p]["t_req"])
            ax.step(t * 1e3, 1 - np.arange(t.size) / t.size, where="post", color=P_COL[p], ls=ls,
                    lw=2 if ls == "-" else 1.2,
                    label=f"{p:.0f} dBm{lab} ({t.size} harmed)" if ls == "-" else None)
    for t, lab in zip(t_values, ["20 µs", "bound", "99 %", "all"]):
        ax.axvline(t * 1e3, color=MUTED, lw=0.8, ls=":")
        ax.text(t * 1e3 * 1.12, 1.4e-3, lab, color=MUTED, fontsize=8, rotation=90, va="bottom")
    ax.plot([], [], color=MUTED, ls=(0, (3, 2)), lw=1.2, label="dashed: carrier-only channel")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("Beacon length T needed to decode (ms)")
    ax.set_ylabel("Share of harmed VSATs still missed")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    save_figure(fig, run_dir, "fig4_treq_ccdf")


def fig_ripple(run_dir, wide_r, narrow_r, wide_harmed, narrow_harmed):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for (rate1, tot), harmed, col, lab in ((wide_r, wide_harmed, INK, "wideband (64 subcarriers)"),
                                         (narrow_r, narrow_harmed, MUTED, "carrier only")):
        rho = 10 * np.log10(rate1[harmed] / tot[harmed])
        rho = np.sort(rho)
        ax.plot(rho, np.arange(1, rho.size + 1) / rho.size, color=col, label=lab,
                lw=2 if col == INK else 1.4)
        print(f"ripple {lab}: median {np.median(rho):+.1f} dB, 1% {np.quantile(rho, .01):+.1f} dB, "
              f"min {rho.min():+.1f} dB")
    ax.axvline(0, color=C[GREEN], lw=1, ls="--")
    ax.text(0.3, 0.1, "‖h‖²/N_tx per sector\n(design assumption)", color=C[GREEN], fontsize=8)
    ax.axvline(-10 * np.log10(3), color=MUTED, lw=1, ls=":")
    ax.text(-10 * np.log10(3) - 0.4, 0.75, "−4.8 dB:\nharm split over\n3 sectors equally",
            color=MUTED, fontsize=8, ha="right")
    ax.set_xlabel("Best-sector beacon gain |wᴴh_s|² relative to Σ_s ‖h_s‖² / N_tx (dB)")
    ax.set_ylabel("CDF over harmed VSATs")
    ax.set_xlim(-30, 10)
    ax.legend(fontsize=8.5, loc="upper left")
    save_figure(fig, run_dir, "fig5_ripple")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--channels-narrow", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--maps", required=True)
    args = ap.parse_args()

    cv = np.load(pathlib.Path(args.curves) / "curves.npz")
    gamma_req_db = float(cv["gamma_req_db"])
    gamma_req, inr_th = 10 ** (gamma_req_db / 10), 10 ** (INR_TH_DB / 10)
    mp = np.load(pathlib.Path(args.maps) / "classes.npz")
    classes, frac, t_values = mp["classes"], mp["fractions"], mp["t_values"]
    run_dir = make_run_dir("step1_report", vars(args) | dict(gamma_req_db=gamma_req_db))

    fig_curves(run_dir, cv, gamma_req_db)
    ch = np.load(pathlib.Path(args.channels) / "channels.npz")
    fig_maps(run_dir, ch, classes, t_values)
    fig_fractions(run_dir, frac, t_values)

    w = zc_precoder()
    wide, wr, wt = per_vsat(ch["hf"], w, gamma_req, inr_th)
    chn = np.load(pathlib.Path(args.channels_narrow) / "channels.npz")
    narrow, nr, nt = per_vsat(chn["h"], w, gamma_req, inr_th)
    fig_treq(run_dir, wide, narrow, t_values)
    fig_ripple(run_dir, (wr, wt), (nr, nt), wide[30.0]["harmed"], narrow[30.0]["harmed"])

    stats = {}
    for p in P_DBM:
        d = wide[p]
        stats[f"{p:.0f}"] = dict(
            harmed_frac=float(d["harmed"].mean()), n_harmed=int(d["harmed"].sum()),
            n_total=int(d["harmed"].size),
            t_req_median_us=float(np.median(d["t_req"]) * 1e6),
            t_req_90_us=float(np.quantile(d["t_req"], 0.9) * 1e6),
            inr_median_db_harmed=float(np.median(10 * np.log10(d["inr"][d["harmed"]]))),
            inr_max_db=float(10 * np.log10(d["inr"].max())),
            no_path_frac=float(np.mean(d["inr"] == 0)))
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
