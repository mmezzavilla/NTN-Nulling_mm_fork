"""Layer A: beacon detection / nonce decoding curves vs total received SNR.

Run from the repo root:  python -m experiments.beacon_curves [--smoke]
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beacon import phy
from beacon.io import make_run_dir, save_figure

B_HZ = 100e6
PFA = 1e-8
PREAMBLE_FRAC = 0.25
TARGET = 0.99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        gamma_db, n_trials, t_values = np.array([15.0, 25.0, 35.0]), 200, {"t20us": 20e-6}
    else:
        gamma_db, n_trials = np.arange(15.0, 32.01, 0.5), 5000
        t_values = {"t20us": 20e-6, "t1ms": 1e-3}

    params = dict(b_hz=B_HZ, pfa=PFA, preamble_frac=PREAMBLE_FRAC, target=TARGET,
                  k_nonce=phy.K_NONCE, n_code=phy.N_CODE, code="5G LDPC BP 20 it, QPSK",
                  n_trials=n_trials, gamma_db=gamma_db.tolist(), t_values=t_values,
                  smoke=args.smoke)
    run_dir = make_run_dir("beacon_curves" + ("_smoke" if args.smoke else ""), params)

    out, psucc = {"gamma_db": gamma_db}, []
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (tag, t_s), ls in zip(t_values.items(), ["-", "--"]):
        t0 = time.time()
        r = phy.simulate_beacon(gamma_db, t_s, B_HZ, n_trials, PFA, seed=0,
                                preamble_frac=PREAMBLE_FRAC)
        print(f"{tag}: {time.time() - t0:.1f} s")
        for k in ("pd", "bler", "psucc"):
            out[f"{k}_{tag}"] = r[k]
        psucc.append(r["psucc"])
        ax.plot(gamma_db, r["pd"], ls, color="#1f77b4", label=f"P_det, {tag}")
        ax.plot(gamma_db, 1 - r["bler"], ls, color="#ff7f0e", label=f"1-BLER, {tag}")
        ax.plot(gamma_db, r["psucc"], ls, color="#2ca02c", label=f"P_succ, {tag}")
    ax.set_xlabel("Total received beacon energy / N0, γ (dB)")
    ax.set_ylabel("Probability")
    ax.grid(True)

    # Save the raw curves first; gamma_req is written only if every T reaches the target.
    np.savez(run_dir / "curves.npz", **out)
    out["gamma_req_db"] = phy.combined_gamma_req_db(gamma_db, psucc, TARGET)
    print(f"gamma_req_db = {out['gamma_req_db']:.2f} dB (max over T)")
    ax.axvline(out["gamma_req_db"], color="grey", ls=":", label="γ_req")
    ax.legend(fontsize=8)
    save_figure(fig, run_dir, "curves")
    np.savez(run_dir / "curves.npz", **out)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
