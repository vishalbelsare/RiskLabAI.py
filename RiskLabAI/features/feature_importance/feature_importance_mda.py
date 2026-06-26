"""
Computes Mean Decrease Accuracy (MDA) feature importance.
"""

from typing import Any

import pandas as pd

from ._common import mda_grouped_importances
from .feature_importance_strategy import FeatureImportanceStrategy


class FeatureImportanceMDA(FeatureImportanceStrategy):
    """
    Computes feature importance using Mean Decrease Accuracy (MDA).

    This method shuffles each feature one by one and measures how
    much the model's performance (e.g., log loss) decreases.
    """

    def __init__(self, classifier: object, n_splits: int = 10, random_state: int = 42):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : object
            An *untrained* scikit-learn classifier.
        n_splits : int, default=10
            Number of splits for cross-validation.
        """
        self.classifier = classifier
        self.n_splits = n_splits
        self.random_state = random_state

    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute MDA feature importance.

        Parameters
        ----------
        x : pd.DataFrame
            The feature data.
        y : pd.Series
            The target data.
        **kwargs : Any
            - 'train_sample_weights': Optional sample weights for training.
            - 'score_sample_weights': Optional sample weights for scoring.

        Returns
        -------
        pd.DataFrame
            DataFrame with "Mean" and "StandardDeviation" of importance.
        """
        # Per-feature MDA = grouped MDA where each feature is its own group.
        groups = {column: [column] for column in x.columns}
        return mda_grouped_importances(
            classifier=self.classifier,
            x=x,
            y=y,
            groups=groups,
            n_splits=self.n_splits,
            random_state=self.random_state,
            train_sample_weights=kwargs.get("train_sample_weights"),
            score_sample_weights=kwargs.get("score_sample_weights"),
        )
