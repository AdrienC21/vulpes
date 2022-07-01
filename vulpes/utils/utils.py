#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""utils.py: Core functions and objects of the Vulpes package.

@Author: Adrien Carrel
"""

import warnings
from typing import List, Any, Union

import numpy as np
import pandas as pd
import xgboost
import lightgbm
from sklearn.utils import all_estimators
from sklearn.metrics import make_scorer, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score, \
    mean_absolute_error, r2_score, mean_squared_error, roc_auc_score, \
    precision_recall_curve, auc, mean_absolute_percentage_error, \
    average_precision_score, pairwise_distances, calinski_harabasz_score, \
    davies_bouldin_score, silhouette_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.extmath import softmax

warnings.filterwarnings("ignore")
# define type Array_like
Array_like = Union[List, pd.DataFrame, pd.Series, np.ndarray, Any]


def pr_auc_score(y: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Function to calculate the PR AUC Score for
    binary and multiclass classification.

    Args:
        y (np.ndarray): True labels.
        y_pred (np.ndarray): Target scores (e.g. probability).
        kwargs: optional keyword arguments of
            sklearn.metrics.precision_recall_curve

    Returns:
        float: Area under the ROC curve.

    Examples:
        >>> from vulpes import pr_auc_score
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> pr_auc_score(y_true, y_scores)
    """
    if len(y_pred.shape) == 1:  # binary classification
        precision, recall, _ = precision_recall_curve(y, y_pred, **kwargs)
    else:
        classes = list(range(y_pred.shape[1]))
        if len(classes) == 2:  # binary classification too
            precision, recall, _ = precision_recall_curve(y,
                                                          y_pred[:, 1],
                                                          **kwargs)
        else:  # multiclass
            Y = label_binarize(y, classes=classes)
            precision, recall, _ = precision_recall_curve(Y.ravel(),
                                                          y_pred.ravel(),
                                                          **kwargs)
    return auc(recall, precision)


def avg_precision(y: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Micro-average precision for binary and
    multiclass.

    Calculate metrics globally by considering each element of the label
    indicator matrix as a label.

    Args:
        y (np.ndarray): True labels.
        y_pred (np.ndarray): Target scores (e.g. probability).
        kwargs: optional keyword arguments of
            sklearn.metrics.average_precision_score

    Returns:
        float: Micro average precision score.

    Examples:
        >>> from vulpes import avg_precision
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> avg_precision(y_true, y_scores)
    """
    if len(y_pred.shape) == 1:  # binary classification
        return average_precision_score(y, y_pred, average="micro", **kwargs)
    classes = list(range(y_pred.shape[1]))
    if len(classes) == 2:  # binary classification too
        return average_precision_score(y, y_pred[:, 1], average="micro",
                                       **kwargs)
    # multiclass
    Y = label_binarize(y, classes=classes)
    return average_precision_score(Y, y_pred, average="micro", **kwargs)


# metrics that will be calculated for each classification models
CUSTOM_SCORER_CLF = {"balanced_accuracy": make_scorer(balanced_accuracy_score,
                                                      greater_is_better=True),
                     "accuracy": make_scorer(accuracy_score,
                                             greater_is_better=True),
                     "precision": make_scorer(precision_score,
                                              average='macro'),
                     "recall": make_scorer(recall_score, average='macro'),
                     "f1": make_scorer(f1_score, average='macro',
                                       greater_is_better=True),
                     "auroc": make_scorer(roc_auc_score, multi_class="ovo",
                                          average="macro",
                                          needs_proba=True,
                                          greater_is_better=True),
                     "auprc": make_scorer(pr_auc_score, needs_proba=True,
                                          greater_is_better=True),
                     "avg_precision": make_scorer(avg_precision,
                                                  needs_proba=True,
                                                  greater_is_better=True)}


def r2_score_adj(y: np.ndarray, y_pred: np.ndarray, *, n: int, p: int,
                 fit_intercept: bool = True) -> float:
    """Calculate Adjusted R2 Score.

    Adapted from:
    https://stackoverflow.com/questions/69901671/
    how-to-create-an-adjusted-r-squared-scorer-using-sklearn-metrics-make-scorer

    Args:
        y (np.ndarray): True labels.
        y_pred (np.ndarray): Target scores (e.g. probability).
        n (int): Number of samples.
        p (int): Number of parameters.
        fit_intercept (bool, optional): Whether of not we fitted the intercept.
            Defaults to True.

    Returns:
        float: Adjusted R2 Score based on the true values and the predicted
            values.

    Examples:
        >>> from vulpes import r2_score_adj
        >>> y_true = np.array([0, 1, 2, 3, 4])
        >>> y_scores = np.array([-0.01, 1.05, 1.98, 3.12, 3.93])
        >>> r2_score_adj(y_true, y_scores, n=5, p=2, fit_intercept=True)
    """
    if fit_intercept:
        rsquared = (1 - np.nansum((y - y_pred) ** 2) /
                    np.nansum((y - np.nanmean(y)) ** 2))
        rsquared_adj = 1 - ((n - 1) / (n - p - 1)) * (1 - rsquared)
    else:
        rsquared = 1 - np.nansum((y - y_pred) ** 2) / np.nansum(y ** 2)
        rsquared_adj = 1 - (n / (n - p)) * (1 - rsquared)
    return rsquared_adj


# metrics that will be calculated for each regression models

# adj r2 scorer is None, will be modified in the
# pipeline below (as we need to retrieve the n and p parameters).
# It's an approximation that doesn't take into account the
# change of n when splitting the dataset
CUSTOM_SCORER_REG = {"r2": make_scorer(r2_score, greater_is_better=True),
                     "rmse": make_scorer(mean_squared_error, squared=True,
                                         greater_is_better=False),
                     "mae": make_scorer(mean_absolute_error,
                                        greater_is_better=False),
                     "mape": make_scorer(mean_absolute_percentage_error,
                                         greater_is_better=False),
                     "adj_r2": None}

# metrics that will be calculated for each clustering algorithms
# Davies–Bouldin Index (DBI), lower is better
CUSTOM_SCORER_CLT = {"calinski_harabasz": calinski_harabasz_score,
                     "silhouette": silhouette_score,
                     "davies_bouldin": davies_bouldin_score}

# Dictionnary with prettier names for the metrics (to print the final result)
METRIC_NAMES = {"balanced_accuracy": "Balanced Accuracy",
                "accuracy": "Accuracy",
                "recall": "Recall",
                "precision": "Precision",
                "f1": "F1 Score",
                "auroc": "AUROC",
                "auprc": "AUPRC",
                "avg_precision": "Micro avg Precision",
                "r2": "R2",
                "rmse": "RMSE",
                "mae": "MAE",
                "mape": "MAPE",
                "adj_r2": "Adjusted R2",
                "calinski_harabasz": "Calinski-Harabasz Index",
                "silhouette": "Mean Silhouette Coefficient",
                "davies_bouldin": "Davies–Bouldin Index"}

# Metrics to reverse (because lower is better)
# ex: rmse, mae, mape
METRICS_TO_REVERSE = ["RMSE",
                      "MAE",
                      "MAPE"]

# Extract classifiers, regressions and clustering algorithms from scikit learn
CLASSIFIERS = [est for est in all_estimators(type_filter=["classifier"])]
REGRESSIONS = [est for est in all_estimators(type_filter=["regressor"])]
CLUSTERING = [est for est in all_estimators(type_filter=["cluster"])]
# Manually add some others
CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
REGRESSIONS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSIONS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
# Remove voting classifiers, multi-task, etc
CLASSIFIERS_TO_REMOVE = set(["CategoricalNB", "ClassifierChain",
                             "ComplementNB", "GaussianProcessClassifier",
                             "MultiOutputClassifier", "MultinomialNB",
                             "NuSVC", "OneVsOneClassifier",
                             "OneVsRestClassifier", "OutputCodeClassifier",
                             "PassiveAggressiveClassifier",
                             "RadiusNeighborsClassifier", "StackingClassifier",
                             "VotingClassifier"])
CLASSIFIERS = [clf for clf in CLASSIFIERS
               if clf[0] not in CLASSIFIERS_TO_REMOVE]
# Remove voting regressions, multi-task, etc
REGRESSIONS_TO_REMOVE = set(["CCA", "IsotonicRegression",
                             "GammaRegressor", "MultiOutputRegressor",
                             "MultiTaskElasticNet", "MultiTaskElasticNetCV",
                             "MultiTaskLasso", "MultiTaskLassoCV",
                             "PoissonRegressor",
                             "PLSCanonical", "PLSRegression",
                             "QuantileRegressor", "RadiusNeighborsRegressor",
                             "RegressorChain", "StackingRegressor",
                             "VotingRegressor"])
REGRESSIONS = [reg for reg in REGRESSIONS
               if reg[0] not in REGRESSIONS_TO_REMOVE]
# Clustering algorithms to remove
CLUSTERING_TO_REMOVE = set(["FeatureAgglomeration"])
CLUSTERING = [clt for clt in CLUSTERING
              if clt[0] not in CLUSTERING_TO_REMOVE]


def sigmoid_(x: float) -> float:
    """Calculate the sigmoid of x

    Args:
        x (float): float

    Returns:
        float: sigmoid(x)

    Examples:

    """
    return 1 / (1 + np.exp(-x))


def sigmoid_array(x: Union[np.ndarray, List]) -> np.ndarray:
    """Based on a list of values, calculate two-class probabilities
    using the sigmoid function.

    Args:
        x (Union[np.ndarray, List]): value (e.g. result
        of the decision function)

    Returns:
        np.ndarray: An array with probabilities for a two class
        classification problem

    Examples:

    """
    return np.array(list(map(lambda e: [sigmoid_(-e), sigmoid_(e)], x)))


def create_model_2(model: Any) -> Any:
    """Create a parent class of the model "model"
    to add a .predict_proba method based on either the
    decision function, or pairwise distances

    Args:
        model (Any): instance of model object

    Returns:
        Any: extended class with a new method

    Examples:
        >>> from vulpes import create_model_2
        >>> from sklearn.neighbors import NearestCentroid
        >>> model = NearestCentroid
        >>> model = create_model_2(model)
        >>> model.fit([[1, 2], [2, 3], [3, 4], [4, 5]], [0, 0, 1, 1])
        >>> model.predict_proba([[2, 2]])
        array([[0.90099855, 0.09900145]])
    """
    if hasattr(model, "decision_function"):
        # if there is a decision function, use it to calculate proba
        class model2(model):
            def __init__(self):
                super().__init__()

            def predict_proba(self, X: Array_like):
                d = self.decision_function(X)
                if len(d.shape) == 1:  # probably two outputs
                    return sigmoid_array(d)
                return softmax(d)
    # no decision function, calculate pairwise distance to clusters instead
    # exemple: NearestCentroid. But need centroids
    else:
        class model2(model):
            def __init__(self):
                super().__init__()

            def predict_proba(self,
                              X: Array_like):
                distances = pairwise_distances(X, Y=self.centroids_,
                                               metric="euclidean",
                                               n_jobs=-1)
                return softmax(-distances)

    return model2()
