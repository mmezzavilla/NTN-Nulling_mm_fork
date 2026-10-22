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


def test_t_required_per_harmed_vsat():
    from beacon.linkbudget import t_required
    rate = np.array([[100., 50, 0], [10., 0, 0], [1., 0, 0]])
    inr = np.array([1., 1., 0.1])
    assert t_required(rate, inr, 256., 0.25).tolist() == pytest.approx([2.56, 25.6])


def test_t_rule_quantile():
    from beacon.linkbudget import t_rule_quantile
    rate = np.array([[256.], [128.], [64.], [32.]])
    inr = np.ones(4)
    assert t_rule_quantile(rate, inr, 256., 0.25, 0.5) == pytest.approx(3.0)   # median of 1,2,4,8


def test_t_rule_raises_on_harmed_vsat_with_zero_beacon_gain():
    rate = np.array([[0., 0, 0]])
    with pytest.raises(ValueError):
        t_rule(rate, np.array([1.0]), 256., 0.25)


def test_class_fractions_indexed_by_class_value():
    from beacon.linkbudget import class_fractions
    c = np.array([[RED, RED, GREEN, GREY]], dtype=np.int8)
    f = class_fractions(c, axis=-1)
    assert f.shape == (1, 3)
    assert f[0, RED] == 0.5 and f[0, GREEN] == 0.25 and f[0, GREY] == 0.25


def test_wideband_single_subcarrier_equals_narrowband():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((4, 3, 64)) + 1j * rng.standard_normal((4, 3, 64))
    w = np.full(64, 1 / 8, complex)
    assert np.allclose(beacon_snr(h[..., None], w, 1., 1e-3, 2.), beacon_snr(h, w, 1., 1e-3, 2.))
    assert np.allclose(inr_worst(h[..., None], 1., 1e8, 2.), inr_worst(h, 1., 1e8, 2.))


def test_wideband_averages_over_subcarriers():
    # Beam gain 1 on one subcarrier and 0 on the other averages to 0.5.
    h = np.zeros((1, 1, 64, 2), complex)
    h[0, 0, 0, 0] = 8.0                      # |h·w*|² = 1 with w = 1/8
    w = np.full(64, 1 / 8, complex)
    psd = noise_psd_w(2.)
    assert beacon_snr(h, w, 1., 1., 2.)[0, 0] == pytest.approx(0.5 / psd)
    assert inr_worst(h, 1., 1e8, 2.)[0] == pytest.approx(32.0 / (psd * 1e8))   # mean ‖h‖² = 64/2


def test_beacon_snr_averages_over_precoder_set():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 3, 64)) + 1j * rng.standard_normal((3, 3, 64))
    w1, w2 = np.exp(1j * rng.uniform(0, 6.3, (2, 64))) / 8
    both = beacon_snr(h, np.stack([w1, w2]), 1., 1e-3, 2.)
    assert np.allclose(both, (beacon_snr(h, w1, 1., 1e-3, 2.) + beacon_snr(h, w2, 1., 1e-3, 2.)) / 2)
    assert np.allclose(beacon_snr(h, w1[None], 1., 1e-3, 2.), beacon_snr(h, w1, 1., 1e-3, 2.))
