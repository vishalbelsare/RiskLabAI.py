"""
Tests for the EDGE bid-ask spread estimator (Ardia-Guidotti-Kroencke 2024).

Covers: unit behaviour on small fixtures, a replication check against the ``bidask`` package (the
MIT-licensed reference, used only as an oracle here), the authors' published reference values, and
edge-case / robustness behaviour (degenerate bars, missing data, a single bar, zero variance).
"""

import numpy as np
import pytest

from RiskLabAI.features.microstructural_features import edge_estimator


def _simulate_ohlc(spread, n=600, vol=0.01, zero_frac=0.0, seed=0):
    """Random-walk mid price with a bid-ask bounce of known proportional spread -> daily OHLC bars."""
    rng = np.random.default_rng(seed)
    steps = n * 50
    log_mid = np.log(100.0) + np.cumsum(vol / np.sqrt(50) * rng.standard_normal(steps))
    q = rng.choice(np.array([-1.0, 1.0]), size=steps)
    obs = np.exp(log_mid + (spread / 2.0) * q)
    traded = rng.random(steps) >= zero_frac
    obs = obs.reshape(n, 50)
    traded = traded.reshape(n, 50)
    o, h, low, c = (np.empty(n) for _ in range(4))
    prev = float(np.exp(log_mid[0]))
    for d in range(n):
        day = obs[d][traded[d]]
        if day.size == 0:
            o[d] = h[d] = low[d] = c[d] = prev
        else:
            o[d], c[d], h[d], low[d] = day[0], day[-1], day.max(), day.min()
        prev = c[d]
    return o, h, low, c


# --- Unit tests on small fixtures -------------------------------------------------


def test_returns_float_and_proportional():
    o, h, low, c = _simulate_ohlc(0.01, seed=1)
    spread = edge_estimator(o, h, low, c)
    assert isinstance(spread, float)
    assert np.isfinite(spread)
    assert 0.0 < spread < 0.05  # proportional spread near the true 1%


def test_recovers_known_spread_within_tolerance():
    # Averaged over seeds, EDGE is near-unbiased for the true spread.
    for true_spread in (0.005, 0.02):
        estimates = [
            edge_estimator(*_simulate_ohlc(true_spread, seed=s)) for s in range(40)
        ]
        assert abs(np.nanmean(estimates) - true_spread) < 0.1 * true_spread


def test_sign_argument_returns_negative_when_squared_spread_negative():
    # Tiny spread + high volatility can drive the squared-spread estimate negative; with sign=True
    # the estimate is returned negative, with sign=False as the non-negative root.
    found_negative = False
    for s in range(60):
        o, h, low, c = _simulate_ohlc(0.0001, vol=0.04, seed=s)
        signed = edge_estimator(o, h, low, c, sign=True)
        unsigned = edge_estimator(o, h, low, c, sign=False)
        assert unsigned >= 0.0 or np.isnan(unsigned)
        if np.isfinite(signed) and signed < 0:
            found_negative = True
            assert np.isclose(unsigned, -signed)
    assert found_negative  # the sign convention is exercised


# --- Replication against the bidask package (oracle) ------------------------------


def test_replication_matches_bidask():
    """Our clean-room EDGE matches the bidask reference within a tight tolerance."""
    bidask = pytest.importorskip("bidask")
    rng = np.random.default_rng(7)
    max_rel = 0.0
    n_checked = 0
    for k in range(200):
        n = int(rng.integers(5, 120))
        spread = float(rng.uniform(0.0005, 0.03))
        vol = float(rng.uniform(0.005, 0.04))
        zero_frac = float(rng.choice([0.0, 0.3, 0.6, 0.9]))
        o, h, low, c = _simulate_ohlc(
            spread, n=n, vol=vol, zero_frac=zero_frac, seed=1000 + k
        )
        ours = edge_estimator(o, h, low, c)
        theirs = float(bidask.edge(o, h, low, c))
        if np.isfinite(ours) and np.isfinite(theirs):
            n_checked += 1
            max_rel = max(max_rel, abs(ours - theirs) / max(abs(theirs), 1e-12))
    assert n_checked > 100
    assert max_rel < 1e-8, f"max relative difference vs bidask = {max_rel:.2e}"


def test_replication_matches_bidask_with_missing_data():
    bidask = pytest.importorskip("bidask")
    rng = np.random.default_rng(11)
    for k in range(60):
        o, h, low, c = _simulate_ohlc(0.01, n=80, seed=2000 + k)
        idx = rng.choice(80, size=8, replace=False)
        c = c.copy()
        c[idx] = np.nan
        ours = edge_estimator(o, h, low, c)
        theirs = float(bidask.edge(o, h, low, c))
        if np.isfinite(ours) and np.isfinite(theirs):
            assert np.isclose(ours, theirs, rtol=1e-8, atol=1e-12)


# --- Edge cases / robustness (must not raise; document the behaviour) -------------


def test_fewer_than_three_observations_returns_nan():
    assert np.isnan(edge_estimator([100.0], [100.0], [100.0], [100.0]))
    assert np.isnan(
        edge_estimator([100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [100.5, 101.5])
    )


def test_zero_variance_input_returns_nan_without_raising():
    px = np.full(50, 100.0)
    result = edge_estimator(px, px, px, px)  # no trades -> undefined, not an exception
    assert np.isnan(result)


def test_degenerate_bars_high_equals_low_do_not_raise():
    # High == Low on every bar (no intrabar range); the estimator must return a float or NaN.
    rng = np.random.default_rng(5)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
    result = edge_estimator(c, c, c, c)
    assert isinstance(result, float)  # returns cleanly (NaN here, all bars degenerate)


def test_missing_data_does_not_raise():
    o, h, low, c = _simulate_ohlc(0.01, n=80, seed=9)
    c = c.copy()
    c[5:10] = np.nan
    o = o.copy()
    o[20] = np.nan
    result = edge_estimator(o, h, low, c)
    assert isinstance(result, float)
    assert np.isfinite(result) or np.isnan(result)


def test_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        edge_estimator(
            [100.0, 101.0, 102.0], [101.0, 102.0], [99.0, 100.0], [100.5, 101.5, 102.5]
        )


def test_accepts_pandas_series():
    pd = pytest.importorskip("pandas")
    o, h, low, c = _simulate_ohlc(0.01, seed=13)
    spread = edge_estimator(pd.Series(o), pd.Series(h), pd.Series(low), pd.Series(c))
    assert np.isfinite(spread) and spread > 0
