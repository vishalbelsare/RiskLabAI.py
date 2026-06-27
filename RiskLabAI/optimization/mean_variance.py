"""
Canonical mean-variance portfolio weight solver.

A single source of truth for the closed-form Markowitz weights used across the
library. Both :func:`RiskLabAI.optimization.nco.get_optimal_portfolio_weights`
and :func:`RiskLabAI.data.denoise.denoising.optimal_portfolio` delegate here so
the inverse-covariance computation is implemented (and hardened) in one place.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
"""

from typing import Optional

import numpy as np


def minimum_variance_weights(
    covariance: np.ndarray, mu: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Compute the closed-form Markowitz portfolio weights.

    If ``mu`` is ``None`` this returns the Global Minimum Variance (GMV)
    portfolio; otherwise it returns the Mean-Variance Optimization (MVO)
    portfolio for the supplied expected-return vector.

    .. math::
        w = \frac{\Sigma^{-1} \mu}{\mathbf{1}^\top \Sigma^{-1} \mu}

    with :math:`\mu = \mathbf{1}` in the GMV case.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix (``N x N``).
    mu : np.ndarray, optional
        Vector of expected returns (``N x 1``). If ``None``, the GMV portfolio
        is computed.

    Returns
    -------
    np.ndarray
        Portfolio weights as an ``N x 1`` column vector, normalized to sum to 1.
    """
    inverse_covariance = np.linalg.inv(covariance)
    ones = np.ones(shape=(inverse_covariance.shape[0], 1))

    if mu is None:
        mu = ones  # GMV portfolio

    weights = inverse_covariance @ mu
    weights /= ones.T @ weights  # normalize weights to sum to 1
    return weights
