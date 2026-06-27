"""
Implements the EDGE bid-ask spread estimator (Ardia, Guidotti & Kroencke 2024).

EDGE estimates the effective bid-ask spread from open, high, low, and close prices. It pools all
four prices and corrects for discrete, infrequently-traded data, giving lower bias and variance than
the close-to-close Roll (1984) estimator and the two-day high-low Corwin-Schultz (2012) estimator,
and it never returns an invalid (negative) point estimate by construction of the sign convention.

Reference:
    Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). Efficient estimation of bid-ask spreads from
    open, high, low, and close prices. Journal of Financial Economics, 161, 103916.
    https://doi.org/10.1016/j.jfineco.2024.103916

This is an original, clean-room implementation written from the published algorithm (the authors'
language-agnostic pseudocode that accompanies the paper), not transcribed from any reference source.
The ``bidask`` package (MIT licence, https://github.com/eguidotti/bidask) is used ONLY as a
replication oracle in the tests to validate this implementation's outputs; none of its code is copied
here.

Admission: EDGE was admitted in Appraisal 03 (see ``library_extension/appraisals/03_verdict.md`` and
``CONTRIBUTIONS_LEDGER.md``, 2026-06-26). The real-data confirmation is a logged follow-up pending an
adequate public intraday / quote dataset (DECISION_LOG 2026-06-26); the admission rests on the
Monte-Carlo ground-truth and held-out evidence.
"""

from __future__ import annotations

import numpy as np

__all__ = ["edge_estimator"]


def edge_estimator(
    open_prices,
    high_prices,
    low_prices,
    close_prices,
    sign: bool = False,
) -> float:
    r"""
    Estimate the effective bid-ask spread with the EDGE estimator (Ardia-Guidotti-Kroencke 2024).

    A single spread estimate is computed from a window of open, high, low, and close prices. The
    returned value is a proportional spread: ``0.01`` corresponds to a 1% spread. The estimator is
    asymptotically unbiased and optimally combines the open-close and high-low information by a
    variance-weighted average of two unbiased moment estimators.

    Preferred-when / avoid-when (regime tag, from the contributions ledger): prefer EDGE over Roll
    and Abdi-Ranaldo for low-frequency spread estimation in all regimes; over Corwin-Schultz at small
    spreads (the edge narrows at very high illiquidity and very large spreads).

    Parameters
    ----------
    open_prices, high_prices, low_prices, close_prices : array-like
        Vectors of open, high, low, and close prices, sorted in ascending order of the timestamp.
        Must share the same length. ``NaN`` entries (missing data) are handled by the estimator.
    sign : bool, default=False
        If True, return a signed estimate (negative when the estimated squared spread is negative);
        if False, return the non-negative square root of the absolute squared-spread estimate.

    Returns
    -------
    float
        The estimated proportional spread, or ``NaN`` when the estimate is undefined (fewer than
        three observations, fewer than two traded periods, or a degenerate / zero-variance input).

    Notes
    -----
    Robustness: the estimator returns ``NaN`` rather than raising on degenerate bars (high equal to
    low), missing data, a single bar, or zero-variance input. It raises ``ValueError`` only when the
    four price vectors differ in length.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    >>> bounce = 1 + 0.005 * rng.choice([-1.0, 1.0], size=500)  # 1% effective spread
    >>> px = mid * bounce
    >>> high = np.maximum(px, mid) * 1.001
    >>> low = np.minimum(px, mid) * 0.999
    >>> spread = edge_estimator(px, high, low, px)
    >>> 0.0 < spread < 0.05
    True

    See Also
    --------
    RiskLabAI.features.microstructural_features.corwin_schultz.corwin_schultz_estimator
        The Corwin-Schultz (2012) two-day high-low baseline.

    References
    ----------
    Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). Efficient estimation of bid-ask spreads from
    open, high, low, and close prices. Journal of Financial Economics, 161, 103916.

    Admitting appraisal: ``library_extension/appraisals/03_verdict.md`` (real-data confirmation is a
    logged follow-up pending an adequate public intraday / quote dataset).
    """
    open_arr = np.asarray(open_prices, dtype=float)
    high_arr = np.asarray(high_prices, dtype=float)
    low_arr = np.asarray(low_prices, dtype=float)
    close_arr = np.asarray(close_prices, dtype=float)

    n = open_arr.shape[0]
    if not (
        high_arr.shape[0] == n and low_arr.shape[0] == n and close_arr.shape[0] == n
    ):
        raise ValueError("open, high, low, and close must have the same length.")
    if n < 3:
        return float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        o = np.log(open_arr)
        h = np.log(high_arr)
        low_log = np.log(low_arr)
        c = np.log(close_arr)
    m = (h + low_log) / 2.0

    def _lag(x: np.ndarray) -> np.ndarray:
        """Shift by one period; the first element (no predecessor) becomes NaN."""
        y = np.full_like(x, np.nan)
        y[1:] = x[:-1]
        return y

    h1 = _lag(h)
    l1 = _lag(low_log)
    c1 = _lag(c)
    m1 = _lag(m)

    # Log-returns. r1 uses no lag, so its first element is masked to NaN to align with the lagged
    # quantities (the first bar has no predecessor and does not enter the estimator).
    r1 = m - o
    r1[0] = np.nan
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    # Trade indicator: the bar traded if its range is non-zero or the low differs from the previous
    # close. NaN where any required input is missing.
    tau_valid = ~(np.isnan(h) | np.isnan(low_log) | np.isnan(c1))
    tau = np.where(tau_valid, ((h != low_log) | (low_log != c1)).astype(float), np.nan)

    def _indicator(condition: np.ndarray, *required: np.ndarray) -> np.ndarray:
        """(tau and condition) as 0/1, NaN where tau or any required input is missing."""
        mask = ~np.isnan(tau)
        for req in required:
            mask = mask & ~np.isnan(req)
        return np.where(mask, ((tau == 1.0) & condition).astype(float), np.nan)

    po1 = _indicator(o != h, o, h)
    po2 = _indicator(o != low_log, o, low_log)
    pc1 = _indicator(c1 != h1, c1, h1)
    pc2 = _indicator(c1 != l1, c1, l1)

    pt = np.nanmean(tau)
    po = np.nanmean(po1) + np.nanmean(po2)
    pc = np.nanmean(pc1) + np.nanmean(pc2)

    if np.nansum(tau) < 2 or po == 0.0 or pc == 0.0:
        return float("nan")

    # De-meaned log-returns, weighted by the trade indicator and trade probability.
    d1 = r1 - np.nanmean(r1) / pt * tau
    d3 = r3 - np.nanmean(r3) / pt * tau
    d5 = r5 - np.nanmean(r5) / pt * tau

    # Two unbiased squared-spread estimators, from the open and from the previous close.
    x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
    x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4

    e1 = np.nanmean(x1)
    e2 = np.nanmean(x2)
    v1 = np.nanmean(x1 * x1) - e1 * e1
    v2 = np.nanmean(x2 * x2) - e2 * e2

    # Variance-weighted average of the two estimators (equal weight if the total variance is not
    # positive), the GMM-optimal combination that minimises estimation variance.
    total_variance = v1 + v2
    if total_variance > 0.0:
        squared_spread = (v2 * e1 + v1 * e2) / total_variance
    else:
        squared_spread = (e1 + e2) / 2.0

    spread = float(np.sqrt(abs(squared_spread)))
    if sign and squared_spread < 0.0:
        spread = -spread
    return spread
