"""Single-LoS checks: ray-traced gNB -> VSAT gain and INR vs a hand link budget.

gNB sector 0 (4x16 tr38901, 10 deg tilt) at [0, 0, 10]; VSAT dish at [500, 0, 10].
Case "main lobe": dish pointed at the gNB (38.1 dBi).
Case "sidelobe": dish pointed 30 deg above the gNB direction, as a VSAT tracking a
satellite would be; gain from the (2 J1(u)/u)^2 dish model at 30 deg off boresight.
The INR uses the pipeline's inr_worst against the hand P·N_tx·g / (kT·NF·B).
Exits with code 1 if any check differs by more than 0.5 dB.
Run from the repo root:  python -m experiments.los_check
"""
import sys

import numpy as np
from scipy.special import jv
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

import SceneConfigSionna  # noqa: F401  (registers the "vsat" antenna pattern)
from beacon.channels import channels_from_cir
from beacon.linkbudget import dbm_to_w, inr_worst

FC = 9.999e9
LAM = 3e8 / FC
D = 500.0
TILT_DEG = 10.0
DISH_GAIN_DBI = 38.1
DISH_DIAMETER = 0.6
OFF_AXIS_DEG = 30.0
P_DBM, B_HZ, NF_DB, N_TX = 30.0, 100e6, 2.0, 64
TOL_DB = 0.5


def dish_gain_db(off_axis_deg):
    if off_axis_deg == 0:
        return DISH_GAIN_DBI
    u = np.pi * DISH_DIAMETER * np.sin(np.deg2rad(off_axis_deg)) / LAM
    return DISH_GAIN_DBI + 10 * np.log10((2 * jv(1, u) / u) ** 2)


def hand_gain_db(off_axis_deg):
    fspl = 20 * np.log10(LAM / (4 * np.pi * D))
    element = 8 - 12 * (TILT_DEG / 65) ** 2  # tr38901 element, 10 deg off the tilted boresight
    return fspl + element + dish_gain_db(off_axis_deg)


def traced_channel(look_at):
    scene = load_scene()
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=4, num_cols=16, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, polarization="V", pattern="tr38901")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, polarization="V", pattern="vsat")
    scene.add(Transmitter(name="tx-0", position=[0.0, 0.0, 10.0],
                          orientation=[0.0, -float(np.deg2rad(TILT_DEG)), 0.0]))
    rx = Receiver(name="vsat", position=[D, 0.0, 10.0])
    scene.add(rx)
    rx.look_at(look_at)
    paths = PathSolver()(scene=scene, max_depth=0, los=True, synthetic_array=True)
    a, _ = paths.cir(normalize_delays=False, out_type="numpy")
    return channels_from_cir(a)  # (1, 1, 64)


def check(name, measured, expected):
    delta = measured - expected
    ok = abs(delta) <= TOL_DB
    print(f"{'PASS' if ok else 'FAIL'} {name}: measured {measured:.2f} dB, hand {expected:.2f} dB, "
          f"delta {delta:+.2f} dB")
    return ok


def main():
    el = np.deg2rad(OFF_AXIS_DEG)
    cases = {"main lobe": ([0.0, 0.0, 10.0], 0.0),
             "sidelobe": ([D - 1000 * np.cos(el), 0.0, 10.0 + 1000 * np.sin(el)], OFF_AXIS_DEG)}
    ok = True
    for name, (target, off_axis) in cases.items():
        h = traced_channel([float(x) for x in target])
        g_meas = 10 * np.log10(np.mean(np.abs(h) ** 2))
        g_hand = hand_gain_db(off_axis)
        ok &= check(f"{name} per-element |h|^2", g_meas, g_hand)
        n0_psd_dbm = -174 + NF_DB
        inr_hand = P_DBM + 10 * np.log10(N_TX) + g_hand - (n0_psd_dbm + 10 * np.log10(B_HZ))
        inr_pipe = 10 * np.log10(inr_worst(h, dbm_to_w(P_DBM), B_HZ, NF_DB)[0])
        ok &= check(f"{name} worst-case INR at {P_DBM:.0f} dBm", inr_pipe, inr_hand)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
