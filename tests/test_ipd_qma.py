"""Targeted regression tests for the IPD-QMA statistical core.

These lock in the high-risk, previously-untested paths flagged by the audit:
DerSimonian-Laird pooling at k=1 / k=2 / k>=3, the HKSJ small-k floor, the
prediction-interval availability rule, the n<10 input guard, the degenerate
lnVR path, and a seeded end-to-end fit.

Anchor values were derived by hand and cross-checked against the
implementation (see PR notes); they are not merely snapshots of current
output.
"""
import contextlib
import io

import numpy as np
import pytest

from ipd_qma import IPDQMA, simulate_location_scale


def _model(**kw):
    kw.setdefault("seed", 12345)
    return IPDQMA(**kw)


# ---------------------------------------------------------------------------
# DerSimonian-Laird pooling
# ---------------------------------------------------------------------------

def test_pool_dl_k1_passthrough_no_pi():
    """k=1: estimate/SE pass through, no heterogeneity, no prediction interval."""
    r = _model()._pool_dl([5.0], [2.0])
    assert r.k == 1
    assert r.estimate == pytest.approx(5.0)
    assert r.se == pytest.approx(2.0)
    assert r.tau2 == 0.0
    assert r.I2 == 0.0
    assert r.Q == 0.0
    assert np.isnan(r.pi_lower) and np.isnan(r.pi_upper)


def test_pool_dl_k1_zero_se_guard():
    """k=1 with SE=0 must not divide by zero; SE is floored, p-value finite."""
    r = _model()._pool_dl([1.0], [0.0])
    assert r.se >= 1e-12
    assert np.isfinite(r.p_value)


def test_pool_dl_k2_tau2_i2_and_no_pi():
    """k=2: DL tau^2/I^2 computed; HKSJ not applied; prediction interval undefined."""
    r = _model(use_hksj=True)._pool_dl([1.0, 3.0], [1.0, 1.0])
    assert r.k == 2
    assert r.estimate == pytest.approx(2.0)
    assert r.tau2 == pytest.approx(1.0)
    assert r.I2 == pytest.approx(50.0)
    # k<3 => no HKSJ, so SE is the plain DL SE = sqrt(1/sum(w_re)) = 1.0
    assert r.se == pytest.approx(1.0)
    assert np.isnan(r.pi_lower) and np.isnan(r.pi_upper)


def test_pool_dl_k3_homogeneous_tau2_zero_with_pi():
    """k>=3 homogeneous data: tau^2=0, I^2=0, and a prediction interval exists."""
    r = _model(use_hksj=True)._pool_dl([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert r.k == 3
    assert r.estimate == pytest.approx(2.0)
    assert r.tau2 == 0.0
    assert r.I2 == 0.0
    assert r.Q == pytest.approx(2.0)
    # q_hksj == 1 here, so HKSJ leaves SE at the DL value sqrt(1/3).
    assert r.se == pytest.approx((1.0 / 3.0) ** 0.5, rel=1e-6)
    assert np.isfinite(r.pi_lower) and np.isfinite(r.pi_upper)
    assert r.pi_lower < r.estimate < r.pi_upper


def test_pool_dl_hksj_floor_never_narrows_below_dl():
    """HKSJ q<1 must be floored to 1 so the CI never narrows below DL.

    est=[1,1.5,2], se=1 => q_hksj=0.25 (<1). Without the max(1,.) floor the
    SE would shrink below the DL SE sqrt(1/3); with the floor it equals it.
    """
    r = _model(use_hksj=True)._pool_dl([1.0, 1.5, 2.0], [1.0, 1.0, 1.0])
    dl_se = (1.0 / 3.0) ** 0.5
    assert r.se == pytest.approx(dl_se, rel=1e-9)
    assert r.se >= dl_se - 1e-12


def test_pool_dl_k_zero_raises():
    with pytest.raises(ValueError):
        _model()._pool_dl([], [])


# ---------------------------------------------------------------------------
# Stage-1 study analysis
# ---------------------------------------------------------------------------

def test_analyze_study_requires_min_n():
    """Fewer than 10 obs/arm must raise, not silently produce junk."""
    m = _model(n_boot=10)
    with pytest.raises(ValueError):
        m.analyze_study(np.arange(5.0), np.arange(12.0))


def test_analyze_study_lnvr_degenerate_is_nan():
    """A constant (zero-variance) arm yields NaN lnVR / SE, not a crash or inf."""
    m = _model(n_boot=10)
    rng = np.random.default_rng(0)
    control = np.ones(20)                 # var_c == 0
    treatment = rng.standard_normal(20)
    res = m.analyze_study(control, treatment, label="degenerate")
    assert np.isnan(res.lnvr)
    assert np.isnan(res.se_lnvr)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_simulate_and_fit_end_to_end_seeded():
    """A small seeded location-scale scenario fits and returns a full profile."""
    studies, labels, true_params = simulate_location_scale(
        K=3, n_range=(30, 40), variance_ratio=2.0, seed=7
    )
    assert len(studies) == 3
    assert true_params["variance_ratio"] == 2.0

    m = IPDQMA(seed=7, n_boot=15)
    with contextlib.redirect_stdout(io.StringIO()):
        results = m.fit(studies, labels)

    assert results["n_studies"] == 3
    profile = results["profile"]
    assert len(profile) == len(m.quantiles)
    for col in ("Quantile", "Effect", "SE", "P", "CI_Lower", "CI_Upper"):
        assert col in profile.columns
    # Pooled SEs are strictly positive and finite.
    assert np.all(np.isfinite(profile["SE"].values))
    assert np.all(profile["SE"].values > 0)
    # Slope test is populated.
    assert "slope_test" in results
    assert np.isfinite(results["slope_test"]["p_value"])
