"""
Computes Clustered Mean Decrease Accuracy (MDA) feature importance.
"""

from typing import Any

import pandas as pd

from ._common import mda_grouped_importances
from .feature_importance_strategy import FeatureImportanceStrategy


class ClusteredFeatureImportanceMDA(FeatureImportanceStrategy):
    """
    Computes clustered feature importance using MDA.

    This method shuffles entire *clusters* of features at a time
    and measures the decrease in model performance.
    """

    def __init__(
        self,
        classifier: object,
        clusters: dict[str, list[str]],
        n_splits: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : object
            An *untrained* scikit-learn classifier.
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster names to lists of feature names.
        n_splits : int, default=10
            Number of splits for cross-validation.
        random_state : int, default=42
            Seed for KFold and shuffling for reproducibility.
        """
        self.classifier = classifier
        self.clusters = clusters
        self.n_splits = n_splits
        self.random_state = random_state

    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute Clustered MDA feature importance.

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
            DataFrame with "Mean" and "StandardDeviation" of importance
            for each *cluster*.
        """
        # Clustered MDA = grouped MDA where each cluster's features are shuffled
        # together (one shared permutation), preserving intra-cluster structure.
        importances_summary = mda_grouped_importances(
            classifier=self.classifier,
            x=x,
            y=y,
            groups=self.clusters,
            n_splits=self.n_splits,
            random_state=self.random_state,
            train_sample_weights=kwargs.get("train_sample_weights"),
            score_sample_weights=kwargs.get("score_sample_weights"),
        )

        importances_summary.index = [f"C_{i}" for i in importances_summary.index]
        return importances_summary
