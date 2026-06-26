"""
Computes Mean Decrease Impurity (MDI) feature importance.
"""

from typing import Any

import pandas as pd
from sklearn.ensemble import BaseEnsemble

from ._common import tree_feature_importances
from .feature_importance_strategy import FeatureImportanceStrategy


class FeatureImportanceMDI(FeatureImportanceStrategy):
    """
    Computes feature importance using Mean Decrease Impurity (MDI).

    This method is specific to tree-based ensembles (like RandomForest)
    and measures importance as the average impurity decrease.
    """

    def __init__(self, classifier: BaseEnsemble):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : BaseEnsemble
            An *untrained* scikit-learn ensemble model (e.g., RandomForestClassifier).
        """
        if not isinstance(classifier, BaseEnsemble):
            raise TypeError("Classifier must be an ensemble (e.g., RandomForest).")

        self.classifier = classifier

    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute MDI feature importance.

        Parameters
        ----------
        x : pd.DataFrame
            The feature data.
        y : pd.Series
            The target data.
        **kwargs : Any
            Keyword arguments for the classifier's `fit` method
            (e.g., `sample_weight`).

        Returns
        -------
        pd.DataFrame
            DataFrame with "Mean" and "StandardDeviation" of importance.
        """
        train_sample_weights = kwargs.get("sample_weight")

        # Fit the ensemble and collect each tree's importances (0 -> NaN).
        importances_df = tree_feature_importances(
            self.classifier, x, y, train_sample_weights
        )

        importances = pd.concat(
            {
                "Mean": importances_df.mean(),
                "StandardDeviation": (
                    importances_df.std() * (importances_df.shape[0] ** -0.5)
                ),
            },
            axis=1,
        )

        # Normalize
        importances /= importances["Mean"].sum()
        return importances
