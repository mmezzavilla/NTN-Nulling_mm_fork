"""Ray-trace gNB -> VSAT channels for the beacon maps (Boulder scene, as in Lambda_CDF_det).

Run from the repo root:  python -m experiments.beacon_channels [--smoke]
"""
import argparse
import time

import numpy as np
from sionna.rt import load_scene

from SceneConfigSionna import SceneConfigSionna
from beacon.channels import channels_from_cir
from beacon.io import make_run_dir

SCENE = "Denver_Scene/Boulder/Boulder.xml"
FC = 9.999e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-drops", type=int, default=100)
    ap.add_argument("--n-vsat", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_drops, args.n_vsat = 1, 5

    params = dict(scene=SCENE, fc=FC, nbs=1, nsect=3, tx_rows=4, tx_cols=16, max_depth=3,
                  tn_rx=0, bs_dist_min=35, bs_dist_max=3000, sat_el_range=[25, 90],
                  n_drops=args.n_drops, n_vsat=args.n_vsat, seed=args.seed, smoke=args.smoke)
    run_dir = make_run_dir("beacon_channels" + ("_smoke" if args.smoke else ""), params)

    scene = load_scene(SCENE)
    h, vsat_pos, bs_pos, az_deg, el_deg = [], [], [], [], []
    extent = point_type = None
    for drop in range(args.n_drops):
        t0 = time.time()
        np.random.seed(args.seed + drop)
        az, el = np.random.uniform(0, 360), np.random.uniform(25, 90)
        cfg = SceneConfigSionna(scene, nbs=1, nsect=3, fc=FC)
        cfg.compute_positions(ntn_rx=args.n_vsat, tn_rx=0, azimuth=az, elevation=el,
                              centerBS=True, bs_dist_min=35, bs_dist_max=3000)
        cfg.compute_paths(tx_rows=4, tx_cols=16, tn_rx_rows=1, tn_rx_cols=1, max_depth=3)
        h.append(channels_from_cir(cfg.a_ntn))
        vsat_pos.append(np.asarray(cfg.rx_ntn_pos))
        bs_pos.append(np.asarray(cfg.tx_pos).reshape(-1)[:3])
        az_deg.append(az)
        el_deg.append(el)
        if drop == 0:
            extent, point_type = np.asarray(cfg.extent), np.asarray(cfg.point_type)
        print(f"drop {drop}: {time.time() - t0:.1f} s, "
              f"{np.count_nonzero(np.any(h[-1] != 0, axis=(1, 2)))}/{args.n_vsat} VSATs with paths",
              flush=True)

    np.savez(run_dir / "channels.npz", h=np.stack(h), vsat_pos=np.stack(vsat_pos),
             bs_pos=np.stack(bs_pos), sat_az_deg=np.array(az_deg), sat_el_deg=np.array(el_deg),
             extent=extent, point_type=point_type)
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
