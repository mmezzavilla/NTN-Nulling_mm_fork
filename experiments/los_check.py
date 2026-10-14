"""Single-LoS check: ray-traced gNB -> VSAT gain vs a hand link budget.

gNB sector 0 (4x16 tr38901, 10 deg tilt) at [0, 0, 10]; VSAT dish at [500, 0, 10]
pointed at the gNB. Exits with code 1 if the two differ by more than 0.5 dB.
Run from the repo root:  python -m experiments.los_check
"""
import sys

import numpy as np
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

import SceneConfigSionna  # noqa: F401  (registers the "vsat" antenna pattern)
from beacon.channels import channels_from_cir

FC = 9.999e9
D = 500.0
TILT_DEG = 10.0
DISH_GAIN_DBI = 38.1
TOL_DB = 0.5


def hand_gain_db():
    lam = 3e8 / FC
    fspl = 20 * np.log10(lam / (4 * np.pi * D))
    element = 8 - 12 * (TILT_DEG / 65) ** 2  # tr38901 element, 10 deg off the tilted boresight
    return fspl + element + DISH_GAIN_DBI


def main():
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
    rx.look_at([0.0, 0.0, 10.0])

    paths = PathSolver()(scene=scene, max_depth=0, los=True, synthetic_array=True)
    a, _ = paths.cir(normalize_delays=False, out_type="numpy")
    h = channels_from_cir(a)[0, 0]
    measured = 10 * np.log10(np.mean(np.abs(h) ** 2))
    expected = hand_gain_db()
    delta = measured - expected
    print(f"measured per-element |h|^2 = {measured:.2f} dB, hand = {expected:.2f} dB, "
          f"delta = {delta:+.2f} dB")
    sys.exit(0 if abs(delta) <= TOL_DB else 1)


if __name__ == "__main__":
    main()
