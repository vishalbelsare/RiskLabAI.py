"""
Shared internals for the feature-importance strategies.

These private helpers factor out logic that the per-feature and clustered
variants of MDI / MDA would otherwise duplicate verbatim:

* :func:`tree_feature_importances` — fit a tree ensemble and collect each
  tree's ``feature_importances_`` (shared by MDI and Clustered-MDI).
* :func:`mda_grouped_importances` — the K-fold permutation-importance loop,
  parameterized by a mapping of *group name -> feature columns* so that the
  per-feature variant (each feature is its own group) and the clustered variant
  (each cluster is a group, shuffled together) run the same code path (shared by
  MDA and Clustered-MDA).

Nothing here is part of the public API.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def tree_feature_importances(
    classifier: Any,
    x: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Fit a tree ensemble and return per-tree feature importances.

    The ensemble is fit in place, then each estimator's
    ``feature_importances_`` becomes one row of the returned frame. Zeros are
    replaced with ``NaN`` (so they are ignored by downstream ``mean``/``std``),
    matching the historical MDI behaviour.

    Parameters
    ----------
    classifier : sklearn ensemble
        An *untrained* tree ensemble (e.g. ``RandomForestClassifier``).
    x : pd.DataFrame
        Feature data.
    y : pd.Series
        Target data.
    sample_weight : np.ndarray, optional
        Sample weights forwarded to ``classifier.fit``.

    Returns
    -------
    pd.DataFrame
        One row per tree, one column per feature; zeros replaced with ``NaN``.
    """
    classifier.fit(x, y, sample_weight=sample_weight)

    importances_dict = {
        i: tree.feature_importances_ for i, tree in enumerate(classifier.estimators_)
    }
    importances_df = pd.DataFrame.from_dict(importances_dict, orient="index")

    if hasattr(classifier, "feature_names_in_"):
        importances_df.columns = classifier.feature_names_in_
    else:
        importances_df.columns = x.columns

    importances_df.replace(0, np.nan, inplace=True)
    return importances_df


def mda_grouped_importances(
    classifier: Any,
    x: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, Sequence[str]],
    n_splits: int,
    random_state: int,
    train_sample_weights: np.ndarray | None = None,
    score_sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    K-fold mean-decrease-accuracy importance for arbitrary feature groups.

    For each fold the classifier is fit and a baseline (negative log-loss) is
    recorded; then, for each group, the group's columns are shuffled *together*
    (one shared row permutation, preserving intra-group structure) and the score
    drop is measured. A group of a single feature reproduces the per-feature MDA;
    a group of several features reproduces the clustered MDA.

    The result index is the group names. Callers that re-label clusters (e.g.
    prefixing ``C_``) should do so on the returned frame.

    Parameters
    ----------
    classifier : sklearn classifier
        An *untrained* classifier exposing ``predict_proba`` and ``classes_``.
    x : pd.DataFrame
        Feature data.
    y : pd.Series
        Target data.
    groups : dict of str -> sequence of str
        Mapping of group name to the feature columns it contains.
    n_splits : int
        Number of K-fold splits (shuffled).
    random_state : int
        Seed for the K-fold split and the per-fold permutations.
    train_sample_weights, score_sample_weights : np.ndarray, optional
        Sample weights for fitting and for scoring. Default to equal weights.

    Returns
    -------
    pd.DataFrame
        Columns ``"Mean"`` and ``"StandardDeviation"``, indexed by group name.
    """
    if train_sample_weights is None:
        train_sample_weights = np.ones(x.shape[0])
    if score_sample_weights is None:
        score_sample_weights = np.ones(x.shape[0])

    cv_generator = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    baseline_scores = pd.Series(dtype=float)
    shuffled_scores = pd.DataFrame(columns=list(groups.keys()), dtype=float)

    for i, (train_idx, test_idx) in enumerate(cv_generator.split(x)):
        logger.debug("Fold %d start ...", i)

        x_train, y_train, w_train = (
            x.iloc[train_idx, :],
            y.iloc[train_idx],
            train_sample_weights[train_idx],
        )
        x_test, y_test, w_test = (
            x.iloc[test_idx, :],
            y.iloc[test_idx],
            score_sample_weights[test_idx],
        )

        fitted_classifier = classifier.fit(X=x_train, y=y_train, sample_weight=w_train)
        pred_proba = fitted_classifier.predict_proba(x_test)
        baseline_scores.loc[i] = -log_loss(
            y_test, pred_proba, labels=classifier.classes_, sample_weight=w_test
        )

        # One RNG per fold so the permutation stream is reproducible and matches
        # the historical per-feature / per-cluster behaviour.
        rng = np.random.default_rng(random_state + i)
        for group_name in shuffled_scores.columns:
            group_cols = list(groups[group_name])

            if not group_cols:  # empty group -> no information removed
                shuffled_scores.loc[i, group_name] = baseline_scores.loc[i]
                continue

            x_test_shuffled = x_test.copy(deep=True)

            # Shuffle the group's columns together with a single row permutation.
            # A copy is required: with pandas copy-on-write the column views are
            # read-only, and shuffling rows jointly preserves intra-group
            # correlation.
            group_data = x_test_shuffled[group_cols].to_numpy(copy=True)
            rng.shuffle(group_data)
            x_test_shuffled[group_cols] = group_data

            shuffled_proba = fitted_classifier.predict_proba(x_test_shuffled)
            shuffled_scores.loc[i, group_name] = -log_loss(
                y_test,
                shuffled_proba,
                labels=classifier.classes_,
                sample_weight=w_test,
            )

    # Importance = drop in score relative to the fold baseline.
    importances = shuffled_scores.rsub(baseline_scores, axis=0)
    importances_summary = pd.concat(
        {
            "Mean": importances.mean(),
            "StandardDeviation": importances.std() * (importances.shape[0] ** -0.5),
        },
        axis=1,
    )
    return importances_summary
