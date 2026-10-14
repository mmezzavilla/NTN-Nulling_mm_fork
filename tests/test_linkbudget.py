import numpy as np
import pytest

from beacon.linkbudget import (GREEN, GREY, RED, beacon_snr, classify, dbm_to_w,
                               inr_worst, noise_psd_w, t_rule, t_rule_bound)


def test_noise_psd():
    assert noise_psd_w(2.0) == pytest.approx(6.30957e-21, rel=1e-5)


def test_dbm_to_w():
    assert dbm_to_w(30.0) == pytest.approx(1.0)


def test_inr_worst_sums_sectors():
    h = np.zeros((1, 3, 64), complex)
    h[0, :, 0] = 1e-6  # ‖h_s‖² = 1e-12 per sector
    assert inr_worst(h, 1.0, 1e8, 2.0)[0] == pytest.approx(4.7547, rel=1e-3)


def test_beacon_snr_hand_value():
    h = np.zeros((1, 3, 64), complex)
    h[0, 0, 0] = 1e-6
    w = np.full(64, 1 / 8, complex)
    assert beacon_snr(h, w, 1.0, 20e-6, 2.0)[0, 0] == pytest.approx(49.528, rel=1e-3)


def test_classify_cases():
    gamma = np.array([[10., 0, 0], [1., 1, 1], [1., 1, 1]])
    inr = np.array([0.1, 1.0, 0.1])
    assert classify(gamma, inr, 5.0, 0.25).tolist() == [GREEN, RED, GREY]


def test_zero_channel_is_grey():
    h = np.zeros((2, 3, 64), complex)
    w = np.full(64, 1 / 8, complex)
    with np.errstate(all="raise"):
        c = classify(beacon_snr(h, w, 1., 1e-3, 2.), inr_worst(h, 1., 1e8, 2.), 256., 0.25)
    assert c.tolist() == [GREY, GREY]


def test_t_rule_picks_worst_harmed():
    rate = np.array([[100., 50, 0], [10., 0, 0], [1., 0, 0]])
    inr = np.array([1., 1., 0.1])  # VSAT 2 is not harmed
    assert t_rule(rate, inr, 256., 0.25) == pytest.approx(25.6)


def test_t_rule_bound_value():
    assert t_rule_bound(256., 64, 3, 10 ** -0.6, 1e8) == pytest.approx(1.9568e-3, rel=1e-3)
