import numpy as np
import pytest

from beacon.phy import detection_threshold, gamma_req_db, measure_pfa, simulate_beacon


def test_threshold():
    assert detection_threshold(1e-8) == pytest.approx(18.4207, rel=1e-4)


def test_measured_pfa_matches():
    # 10⁻⁸ is unmeasurable; the same threshold formula is checked at 10⁻².
    assert measure_pfa(20e-6, 1e8, 1e-2, 20000, seed=1) == pytest.approx(1e-2, abs=3e-3)


def test_high_snr_always_succeeds():
    assert simulate_beacon(np.array([40.0]), 20e-6, 1e8, 200)["psucc"][0] == 1.0


def test_low_snr_fails():
    assert simulate_beacon(np.array([0.0]), 20e-6, 1e8, 200)["psucc"][0] < 0.05


def test_too_short_beacon_raises():
    with pytest.raises(ValueError):
        simulate_beacon(np.array([30.]), 2e-6, 1e8, 10)  # 200 samples


def test_gamma_req_interpolates():
    assert gamma_req_db(np.array([20., 21, 22]), np.array([0.5, 0.98, 1.0])) == pytest.approx(21.5)


def test_gamma_req_unreached_raises():
    with pytest.raises(ValueError):
        gamma_req_db(np.array([20., 21]), np.array([0.5, 0.9]))
