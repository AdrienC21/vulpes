#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main.py: Core functions and objects of the Vulpes package.

@Author: Adrien Carrel
@Credit: Adrien Carrel
"""

import numbers
from time import perf_counter
from typing import List, Dict, Any, Union, Tuple
from collections import defaultdict
from collections.abc import Iterable
from os.path import join as opj

import numpy as np
import pandas as pd
import xgboost
import lightgbm
from tqdm import tqdm
from sklearn.utils import all_estimators
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, RepeatedKFold, \
    StratifiedShuffleSplit, _validation, train_test_split, TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score, \
    mean_absolute_error, r2_score, mean_squared_error, roc_auc_score, \
    precision_recall_curve, auc, mean_absolute_percentage_error, \
    average_precision_score, pairwise_distances, calinski_harabasz_score, \
    davies_bouldin_score, silhouette_score
from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
from sklearn.utils.extmath import softmax
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError


def pr_auc_score(y, y_pred, **kwargs) -> float:
    """Function to calculate the PR AUC Score for
    binary and multiclass classification.

    Args:
        y (ndarray): True labels.
        y_pred (ndarray): Target scores (e.g. probability).
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


def avg_precision(y, y_pred, **kwargs) -> float:
    """Micro-average precision for binary and
    multiclass.

    Calculate metrics globally by considering each element of the label
    indicator matrix as a label.

    Args:
        y (ndarray): True labels.
        y_pred (ndarray): Target scores (e.g. probability).
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
CUSTOM_SCORER_CLF = {"accuracy": make_scorer(accuracy_score,
                                             greater_is_better=True),
                     "balanced_accuracy": make_scorer(balanced_accuracy_score,
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


def r2_score_adj(y, y_pred, *, n: int, p: int,
                 fit_intercept: bool = True) -> float:
    """Calculate Adjusted R2 Score.

    Adapted from:
    https://stackoverflow.com/questions/69901671/
    how-to-create-an-adjusted-r-squared-scorer-using-sklearn-metrics-make-scorer

    Args:
        y (ndarray): True labels.
        y_pred (ndarray): Target scores (e.g. probability).
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
METRIC_NAMES = {"accuracy": "Accuracy",
                "balanced_accuracy": "Balanced Accuracy",
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
classifiers_to_remove = set(["ClassifierChain", "GaussianProcessClassifier",
                             "MultinomialNB", "MultiOutputClassifier",
                             "NuSVC", "OneVsOneClassifier",
                             "OneVsRestClassifier", "OutputCodeClassifier",
                             "PassiveAggressiveClassifier",
                             "StackingClassifier", "VotingClassifier",
                             "RadiusNeighborsClassifier", "CategoricalNB"])
CLASSIFIERS = [clf for clf in CLASSIFIERS
               if clf[0] not in classifiers_to_remove]
# Remove voting regressions, multi-task, etc
regressions_to_remove = set(["CCA", "IsotonicRegression", "GammaRegressor",
                             "MultiOutputRegressor", "MultiTaskElasticNet",
                             "MultiTaskElasticNetCV", "MultiTaskLasso",
                             "MultiTaskLassoCV", "PLSCanonical",
                             "PLSRegression", "QuantileRegressor",
                             "RadiusNeighborsRegressor", "RegressorChain",
                             "StackingRegressor", "VotingRegressor"])
REGRESSIONS = [reg for reg in REGRESSIONS
               if reg[0] not in regressions_to_remove]
# Clustering algorithms to remove
clustering_to_remove = set(["FeatureAgglomeration"])
CLUSTERING = [clt for clt in CLUSTERING
              if clt[0] not in clustering_to_remove]


def create_model_2(model):
    """Create a parent class of the model "model"
    to add a .predict_proba method based on either the
    decision function, or pairwise distances

    Args:
        model (type): instance of model object

    Returns:
        class: extended class with a new method

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

            def predict_proba(self, X):
                d = self.decision_function(X)
                return softmax(d)
    # no decision function, calculate pairwise distance to clusters instead
    # exemple: NearestCentroid. But need centroids
    else:
        class model2(model):
            def __init__(self):
                super().__init__()

            def predict_proba(self, X):
                distances = pairwise_distances(X, Y=self.centroids_,
                                               metric="euclidean",
                                               n_jobs=-1)
                return softmax(-distances)

    return model2()


class CoreVulpes():
    """
    Parent class with shared methods for the classes
    Classifiers and Regressions
    """

    def __init__(self):
        # store the metrics dataframe after fitting
        self.df_models = None  # all the models
        self.df_best_model = None  # the best one with a voting clf/reg
        # store the fitted models
        self.fitted_models_ = {}  # all the models
        self.best_model_ = None  # the best one with a voting clf/reg

    def predefined_preprocessing(self):
        if isinstance(self.preprocessing, str):
            if self.preprocessing == "default":
                # Imputer + standard scaler for not categorical values
                numeric_transformer = Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())]
                )
                # OneHotEncoder for categorical values
                categorical_transformer = OneHotEncoder(
                    handle_unknown="ignore")

                preprocessing = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer,
                         selector(dtype_exclude=["category", object])),
                        ("cat", categorical_transformer,
                         selector(dtype_include=["category", object])),
                    ]
                )

                preprocessing_pipeline = Pipeline(steps=[("preprocessing",
                                                          preprocessing)])
            # Insert predefined preprocessing pipelines here!
            else:
                raise ValueError("Unknown parameter: preprocessing."
                                 "Please enter a valid preprocessing "
                                 "(an already implemented one, "
                                 ", a Pipeline object, or None).")
        # if it's not a string, not a pipeline, or not None, raise error
        elif (not(isinstance(self.preprocessing, Pipeline))
              or not(self.preprocessing is None)):
            raise TypeError("Preprocessing must be a string, "
                            "a Pipeline object, or None.")
        else:  # return the given preprocessing pipeline or None object
            preprocessing_pipeline = self.preprocessing
        return preprocessing_pipeline

    def create_pipeline(self, model_name: str, model) -> Pipeline:
        # Preprocessing
        preprocessing_pipeline = self.predefined_preprocessing()

        # Model
        model_pipeline = Pipeline(steps=[(model_name, model)])

        # Merge all the steps
        pipelines = [preprocessing_pipeline,
                     model_pipeline]
        steps = [p.steps for p in pipelines if not(p is None)]
        return Pipeline(steps=steps)

    def predefined_cv(self, X=None, groups=None) -> Any:
        if self.cv == "default":
            # if groups, cross validation is a
            # stratified shuffle, else,
            # a repeatedKFold
            if groups is None:
                cv = RepeatedKFold(n_splits=5, n_repeats=5,
                                   random_state=self.random_state)
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=5,
                    test_size=self.test_size,
                    random_state=self.random_state)
                cv = sss.split(X, groups)
        elif self.cv == "timeseries":
            cv = TimeSeriesSplit(n_splits=5, test_size=self.test_size)

        # Insert predefined cross validation here!
        else:
            raise ValueError(
                f"Unknown cross validation: {self.cv}")
        return cv

    def predefined_list_models(
        self,
        models_to_try: Union[str, List[Tuple[str, Any]]] = "all"
    ) -> List[Tuple[str, Any]]:
        if not(isinstance(models_to_try, str)):
            return models_to_try
        # else, it's a string, search for predefined list of models
        if models_to_try == "all":
            if self.task == "classification":
                return CLASSIFIERS
            elif self.task == "regression":
                return REGRESSIONS
            elif self.task == "clustering":
                return CLUSTERING
        # Insert new predefined list here!
        else:
            raise ValueError(
                f'Unknown parameter models_to_try: {models_to_try}. '
                'Please enter a valid list of models (tuple like '
                '("XGBClassifier", xgboost.XGBClassifier)) or an '
                'existing predefined list of models("all", ...)')

    def build_best_models(self, X, y, *, sample_weight=None, groups=None,
                          nb_models: int = 5, sort_result_by: str = None,
                          ascending: bool = None,
                          voting: str = "hard", weights=None) -> pd.DataFrame:
        if self.task == "clustering":
            raise ValueError("Can't create a Voting Clustering algorithm.")
        if self.df_models is None:
            raise NotFittedError(
                "Please fit the models first by calling the method .fit "
                "before building an ensemble model")
        if nb_models > len(self.df_models):
            raise ValueError(
                f"Not enough trained models ({len(self.df_models)}) "
                f"to select the best {nb_models} ones")

        top = perf_counter()  # start to measure fitting time

        # if undefined, take default values for sort_result_by and ascending
        # to select the best models
        if sort_result_by is None:
            sort_result_by = self.sort_result_by
        if ascending is None:
            ascending = self.ascending
        # sort the models based on the given metric
        sorted_df_models = (self.df_models
                            .sort_values(by=sort_result_by,
                                         ascending=ascending))
        # name of the best models
        best_models_names = list(sorted_df_models.index)[:nb_models]
        # dictionnary: model name -> corresponding class
        dict_models = dict(self.predefined_list_models(self.models_to_try))
        # list of tuple (model name, instance of model)

        # create the list of estimators
        if self.task == "classification":  # check predict_proba method
            estimators = []
            for b in best_models_names:
                if not(hasattr(dict_models[b], "predict_proba")):
                    estimator = create_model_2(dict_models[b])
                else:
                    estimator = dict_models[b]()
                estimators.append((b.lower(), estimator))
        else:
            estimators = [(b.lower(), dict_models[b]())
                          for b in best_models_names]

        # define the voting model
        if self.task == "classification":
            voting = VotingClassifier(estimators, voting=voting,
                                      weights=weights, n_jobs=-1)
        elif self.task == "regression":
            voting = VotingRegressor(estimators, weights=weights,
                                     n_jobs=-1)
        else:
            raise NotImplementedError("Unknown task: {self.task}")
        voting_name = voting.__class__.__name__.lower()
        # Add preprocessing, create a pipeline
        pipe = self.create_pipeline(voting_name,
                                    voting)
        # adjust some hyperparameters when available for each model inside the
        # voting model
        model_params = {}
        for (model_name, model_instance) in estimators:
            model_available_params = model_instance.get_params().keys()
            if "random_state" in model_available_params:
                model_params[f"{voting_name}__{model_name}__"
                             "random_state"] = self.random_state
            if "n_jobs" in model_available_params:
                model_params[f"{voting_name}__{model_name}__n_jobs"] = -1
            if "probability" in model_available_params:
                model_params[f"{voting_name}__{model_name}__"
                             "probability"] = True
            # change the loss to allow multiclass and predict proba
            if model_name == "SGDClassifier".lower():
                model_params[f"{voting_name}__{model_name}__loss"] = "log"
        if model_params != {}:
            pipe.set_params(**model_params)

        ######
        # Voting Classifier / Regressor don't really support sample weight
        # all of the models need a sample weight :/
        fit_params = {}
        ######

        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        if self.use_cross_validation:
            if not(hasattr(self.cv, "split") or
                    isinstance(self.cv, numbers.Integral) or
                    isinstance(self.cv, Iterable) or
                    isinstance(self.cv, str)):
                raise ValueError(
                    "Expected cv as an integer, cross-validation "
                    "object (from sklearn.model_selection), "
                    "iterable or valid string predefined cv"
                )
            if isinstance(self.cv, str):
                cv = self.predefined_cv(X, groups)
            else:
                cv = self.cv
            try:
                cv_model = cross_validate(
                    pipe, X, y, cv=cv,
                    return_estimator=True, n_jobs=-1,
                    fit_params=fit_params,
                    scoring=self.custom_scorer)
            except ValueError as e:
                print(e)
                print("Cross validation failed. If groups provided, ")
                print("maybe can't stratify because some groups "
                      "represented in the dataset don't contains enough"
                      " samples.")
                print("Using RepeatedKFold instead.")
                cv = RepeatedKFold(n_splits=5, n_repeats=5,
                                   random_state=self.random_state)
                cv_model = cross_validate(
                    pipe, X, y, cv=cv,
                    return_estimator=True, n_jobs=-1,
                    fit_params=fit_params,
                    scoring=self.custom_scorer)
            except Exception as e:
                print(e)

            # add metrics to the lists

            # nan mean are used, because when folding,
            # a label can be missing
            # in a particular fold, thus creating nan values
            for metric_name in self.custom_scorer.keys():
                print_metric_name = METRIC_NAMES.get(metric_name,
                                                     metric_name)
                (metrics_dic[print_metric_name]
                 .append(np.nanmean(cv_model[f"test_{metric_name}"])))

            metrics_dic["Model"].append(f"Voting ({nb_models}-best)")
            # add running time
            metrics_dic["Running time"].append(perf_counter() - top)
        else:
            try:
                X_train, X_test, y_train, \
                    y_test = train_test_split(
                        X, y, test_size=self.test_size,
                        shuffle=self.shuffle,
                        stratify=groups, random_state=self.random_state)
                pipe.fit(X_train, y_train, **fit_params)
                res = _validation._score(pipe, X_test, y_test,
                                         self.custom_scorer,
                                         error_score="raise")
            except Exception as e:
                raise RuntimeError(f"Error when fitting: {e}")

            # add metrics to the lists

            # nan mean are used, because when folding,
            # a label can be missing
            # in a particular fold, thus creating nan values
            for metric_name in self.custom_scorer.keys():
                print_metric_name = METRIC_NAMES.get(metric_name,
                                                     metric_name)
                (metrics_dic[print_metric_name]
                 .append(np.nanmean(res[metric_name])))
            metrics_dic["Model"].append(f"Voting ({nb_models}-best)")
            # add running time
            metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                metrics_dic[metric_name] = [-x for x in
                                            metrics_dic[metric_name]]

        # Metric for the fitted "best model"
        df_best_model = pd.DataFrame.from_dict(metrics_dic)
        df_best_model = df_best_model.set_index("Model")
        self.best_model_ = pipe
        self.df_best_model = df_best_model

        return df_best_model

    def get_fitted_models(self):
        if self.fitted_models_ is None:
            raise NotFittedError(
                "Fit some models before retrieving them by calling "
                "the method .fit")
        return self.fitted_models_

    def get_best_model(self):
        if self.best_model_ is None:
            if self.fitted_models_ is None:
                raise NotFittedError(
                    "Fit some models before retrieving them by calling "
                    "the method .fit")
            raise NotFittedError(
                "Many models have been fitted. But the 'best model' "
                "hasn't been fitted yet. Please call the method "
                ".build_best_models before retrieving it.")
        return self.best_model_


class Classifiers(CoreVulpes):
    def __init__(self, *,
                 models_to_try: Union[str, List[Tuple[str, Any]]] = "all",
                 custom_scorer: Dict[str, Any] = CUSTOM_SCORER_CLF,
                 preprocessing: Union[Pipeline, str] = None,
                 use_cross_validation: bool = True,
                 cv: Any = "default",
                 test_size: float = 0.2,
                 shuffle: bool = False,
                 sort_result_by: str = "Balanced Accuracy",
                 ascending: bool = False,
                 save_results: bool = False,
                 path_results: str = "",
                 random_state: int = 42):
        super().__init__()
        self.task = "classification"
        self.models_to_try = self.predefined_list_models(models_to_try)
        self.custom_scorer = custom_scorer
        self.preprocessing = preprocessing
        self.use_cross_validation = use_cross_validation
        self.cv = cv
        self.test_size = test_size
        self.shuffle = shuffle
        self.sort_result_by = sort_result_by
        self.ascending = ascending
        self.save_results = save_results
        self.path_results = path_results
        self.random_state = random_state

    def fit(self, X, y, *, sample_weight=None, groups=None) -> pd.DataFrame:
        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        # Fit and calculate metrics for all the models
        for name, model in tqdm(self.models_to_try):
            print(name)
            top = perf_counter()

            # extend the class with a custom predict proba function
            # based on a decision function or pairwise distances
            if not(hasattr(model, "predict_proba")):
                model = create_model_2(model)
            model_name = name.lower()

            # Create the pipeline
            pipe = self.create_pipeline(model_name, model)

            # When available:
            # - Activate "probability" parameter to calculate more metrics
            # - maximize n_jobs
            # - fix a random_state
            # - change the loss for SGDClassifier to allow the
            # predict_proba method
            model_params = {}
            model_available_params = model().get_params().keys()
            if "random_state" in model_available_params:
                model_params[f"{model_name}__random_state"] = self.random_state
            if "n_jobs" in model_available_params:
                model_params[f"{model_name}__n_jobs"] = -1
            if "probability" in model_available_params:
                model_params[f"{model_name}__probability"] = True
            # change the loss to allow multiclass and predict proba
            if name == "SGDClassifier":
                model_params[f"{model_name}__loss"] = "log"
            if model_params != {}:
                pipe.set_params(**model_params)

            # if sample_weight is a parameter of the fit function, update
            # fit parameters
            if not("sample_weight" in model.fit.__code__.co_varnames):
                fit_params = {}
            else:
                fit_params = {f"{model_name}__sample_weight": sample_weight}

            if self.use_cross_validation:
                if not(hasattr(self.cv, "split") or
                       isinstance(self.cv, numbers.Integral) or
                       isinstance(self.cv, Iterable) or
                       isinstance(self.cv, str)):
                    raise ValueError(
                        "Expected cv as an integer, cross-validation "
                        "object (from sklearn.model_selection), "
                        "iterable or valid string predefined cv"
                    )
                if isinstance(self.cv, str):
                    cv = self.predefined_cv(X, groups)
                else:
                    cv = self.cv
                try:
                    cv_model = cross_validate(
                        pipe, X, y, cv=cv,
                        return_estimator=True, n_jobs=-1,
                        fit_params=fit_params,
                        scoring=self.custom_scorer)
                except ValueError as e:
                    print(e)
                    print("Cross validation failed. If groups provided, ")
                    print("maybe can't stratify because some groups "
                          "represented in the dataset don't contains enough"
                          " samples.")
                    print("Using RepeatedKFold instead.")
                    cv = RepeatedKFold(n_splits=5, n_repeats=5,
                                       random_state=self.random_state)
                    cv_model = cross_validate(
                        pipe, X, y, cv=cv,
                        return_estimator=True, n_jobs=-1,
                        fit_params=fit_params,
                        scoring=self.custom_scorer)
                except Exception as e:
                    print(e)

                # store fitted models
                self.fitted_models_[name] = cv_model["estimator"]

                # add metrics to the lists

                # nan mean are used, because when folding,
                # a label can be missing
                # in a particular fold, thus creating nan values
                for metric_name in self.custom_scorer.keys():
                    print_metric_name = METRIC_NAMES.get(metric_name,
                                                         metric_name)
                    (metrics_dic[print_metric_name]
                     .append(np.nanmean(cv_model[f"test_{metric_name}"])))
                metrics_dic["Model"].append(name)  # add name
                # add running time
                metrics_dic["Running time"].append(perf_counter() - top)
            else:
                try:
                    X_train, X_test, y_train, \
                        y_test = train_test_split(
                            X, y, test_size=self.test_size,
                            shuffle=self.shuffle,
                            stratify=groups, random_state=self.random_state)
                    pipe.fit(X_train, y_train, **fit_params)
                    res = _validation._score(pipe, X_test, y_test,
                                             self.custom_scorer,
                                             error_score="raise")
                except Exception as e:
                    raise RuntimeError(f"Error when fitting: {e}")

                # store fitted models
                self.fitted_models_[name] = pipe

                # add metrics to the lists

                # nan mean are used, because when folding,
                # a label can be missing
                # in a particular fold, thus creating nan values
                for metric_name in self.custom_scorer.keys():
                    print_metric_name = METRIC_NAMES.get(metric_name,
                                                         metric_name)
                    (metrics_dic[print_metric_name]
                     .append(np.nanmean(res[metric_name])))
                metrics_dic["Model"].append(name)  # add name
                # add running time
                metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                metrics_dic[metric_name] = [-x for x in
                                            metrics_dic[metric_name]]
        # create a dataframe with the results
        df_models = pd.DataFrame.from_dict(metrics_dic)
        df_models = (df_models
                     .sort_values(by=self.sort_result_by,
                                  ascending=self.ascending)
                     .set_index("Model"))
        self.df_models = df_models

        if self.save_results:  # save results
            df_models.to_csv(opj.path.join(self.path_results,
                                           f"vulpes_results_{self.task}.csv"))
        return df_models


class Regressions(CoreVulpes):
    def __init__(self, *,
                 models_to_try: Union[str, List[Tuple[str, Any]]] = "all",
                 custom_scorer: Dict[str, Any] = CUSTOM_SCORER_REG,
                 preprocessing: Union[Pipeline, str] = None,
                 use_cross_validation: bool = True,
                 cv: Any = "default",
                 test_size: float = 0.2,
                 shuffle: bool = True,
                 sort_result_by: str = "MAPE",
                 ascending: bool = True,
                 save_results: bool = False,
                 path_results: str = "",
                 random_state: int = None):
        super().__init__()
        self.task = "regression"
        self.models_to_try = self.predefined_list_models(models_to_try)
        self.custom_scorer = custom_scorer
        self.preprocessing = preprocessing
        self.use_cross_validation = use_cross_validation
        self.cv = cv
        self.test_size = test_size
        self.shuffle = shuffle
        self.sort_result_by = sort_result_by
        self.ascending = ascending
        self.save_results = save_results
        self.path_results = path_results
        self.random_state = random_state

    def fit(self, X, y, *, sample_weight=None, groups=None) -> pd.DataFrame:
        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        # add adjusted r2 to the custom_scorer
        # we will fix fit_intercept to True
        if "adj_r2" in self.custom_scorer:
            self.custom_scorer["adj_r2"] = make_scorer(r2_score_adj,
                                                       n=X.shape[0],
                                                       p=X.shape[1],
                                                       fit_intercept=True,
                                                       greater_is_better=True)

        # Fit and calculate metrics for all the models
        for name, model in tqdm(self.models_to_try):
            print(name)
            top = perf_counter()

            # Create the pipeline
            pipe = self.create_pipeline(name.lower(), model)

            # When available:
            # - maximize n_jobs
            # - fix a random_state
            model_params = {}
            model_available_params = model().get_params().keys()
            if "random_state" in model_available_params:
                model_params[(f"{name.lower()}__"
                              "random_state")] = self.random_state
            if "n_jobs" in model_available_params:
                model_params[f"{name.lower()}__n_jobs"] = -1
            if model_params != {}:
                pipe.set_params(**model_params)

            # if sample_weight is a parameter of the fit function, update
            # fit parameters
            if not("sample_weight" in model.fit.__code__.co_varnames):
                fit_params = {}
            else:
                fit_params = {f"{name.lower()}__sample_weight": sample_weight}

            if self.use_cross_validation:
                if not(hasattr(self.cv, "split") or
                       isinstance(self.cv, numbers.Integral) or
                       isinstance(self.cv, Iterable) or
                       isinstance(self.cv, str)):
                    raise ValueError(
                        "Expected cv as an integer, cross-validation "
                        "object (from sklearn.model_selection), "
                        "iterable or valid string predefined cv"
                    )
                if isinstance(self.cv, str):
                    cv = self.predefined_cv(X, groups)
                else:
                    cv = self.cv

                try:
                    cv_model = cross_validate(
                        pipe, X, y, cv=cv,
                        return_estimator=True, n_jobs=-1,
                        fit_params=fit_params,
                        scoring=self.custom_scorer)
                except ValueError as e:
                    print(e)
                    print("Cross validation failed. If groups provided, ")
                    print("maybe can't stratify because some groups "
                          "represented in the dataset don't contains enough"
                          " samples.")
                    print("Using RepeatedKFold instead.")
                    cv = RepeatedKFold(n_splits=5, n_repeats=5,
                                       random_state=self.random_state)
                    cv_model = cross_validate(
                        pipe, X, y, cv=cv,
                        return_estimator=True, n_jobs=-1,
                        fit_params=fit_params,
                        scoring=self.custom_scorer)
                except Exception as e:
                    print(e)

                # store fitted models
                self.fitted_models_[name] = cv_model["estimator"]

                # add metrics to the lists

                # nan mean are used, because when folding,
                # a label can be missing
                # in a particular fold, thus creating nan values
                for metric_name in self.custom_scorer.keys():
                    print_metric_name = METRIC_NAMES.get(metric_name,
                                                         metric_name)
                    (metrics_dic[print_metric_name]
                     .append(np.nanmean(cv_model[f"test_{metric_name}"])))
                metrics_dic["Model"].append(name)  # add name
                # add running time
                metrics_dic["Running time"].append(perf_counter() - top)
            else:
                try:
                    X_train, X_test, y_train, \
                        y_test = train_test_split(
                            X, y, test_size=self.test_size,
                            shuffle=self.shuffle,
                            stratify=groups, random_state=self.random_state)
                    pipe.fit(X_train, y_train, **fit_params)
                    res = _validation._score(pipe, X_test, y_test,
                                             self.custom_scorer,
                                             error_score="raise")
                except Exception as e:
                    raise RuntimeError(f"Error when fitting: {e}")

                # store fitted models
                self.fitted_models_[name] = pipe

                # add metrics to the lists

                # nan mean are used, because when folding,
                # a label can be missing
                # in a particular fold, thus creating nan values
                for metric_name in self.custom_scorer.keys():
                    print_metric_name = METRIC_NAMES.get(metric_name,
                                                         metric_name)
                    (metrics_dic[print_metric_name]
                     .append(np.nanmean(res[metric_name])))
                metrics_dic["Model"].append(name)  # add name
                # add running time
                metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                metrics_dic[metric_name] = [-x for x in
                                            metrics_dic[metric_name]]
        df_models = pd.DataFrame.from_dict(metrics_dic)
        df_models = (df_models
                     .sort_values(by=self.sort_result_by,
                                  ascending=self.ascending)
                     .set_index("Model"))
        self.df_models = df_models

        if self.save_results:  # save results
            df_models.to_csv(opj.path.join(self.path_results,
                                           f"vulpes_results_{self.task}.csv"))
        return df_models


class Clustering(CoreVulpes):
    def __init__(self, *,
                 models_to_try: Union[str, List[Tuple[str, Any]]] = "all",
                 custom_scorer: Dict[str, Any] = CUSTOM_SCORER_REG,
                 preprocessing: Union[Pipeline, str] = None,
                 nb_clusters: int = 3,
                 min_samples: int = 5,
                 eps: int = 0.5,
                 sort_result_by: str = "Davies–Bouldin Index",
                 ascending: bool = True,
                 save_results: bool = False,
                 path_results: str = "",
                 random_state: int = None):
        super().__init__()
        self.task = "clustering"
        self.models_to_try = self.predefined_list_models(models_to_try)
        self.custom_scorer = custom_scorer
        self.preprocessing = preprocessing
        self.nb_clusters = nb_clusters
        self.min_samples = min_samples
        self.eps = eps
        self.sort_result_by = sort_result_by
        self.ascending = ascending
        self.save_results = save_results
        self.path_results = path_results
        self.random_state = random_state

    def fit(self, X, y, *, sample_weight=None, groups=None) -> pd.DataFrame:
        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        # Fit and calculate metrics for all the models
        for name, model in tqdm(self.models_to_try):
            print(name)
            top = perf_counter()

            # Create the pipeline
            pipe = self.create_pipeline(name.lower(), model)

            # When available:
            # - maximize n_jobs
            # - fix a random_state
            # - nb_clusters
            # - min_sample
            # - epsilon (eps)
            model_params = {}
            model_available_params = model().get_params().keys()
            if "random_state" in model_available_params:
                model_params[(f"{name.lower()}__"
                              "random_state")] = self.random_state
            if "n_jobs" in model_available_params:
                model_params[f"{name.lower()}__n_jobs"] = -1
            if "nb_clusters" in model_available_params:
                model_params[f"{name.lower()}__nb_clusters"] = self.nb_clusters
            if "min_samples" in model_available_params:
                model_params[f"{name.lower()}__min_samples"] = self.min_samples
            if "eps" in model_available_params:
                model_params[f"{name.lower()}__eps"] = self.eps
            if model_params != {}:
                pipe.set_params(**model_params)

            # if sample_weight is a parameter of the fit function, update
            # fit parameters
            if not("sample_weight" in model.fit.__code__.co_varnames):
                fit_params = {}
            else:
                fit_params = {f"{name.lower()}__sample_weight": sample_weight}

            try:
                pipe.fit(X, **fit_params)
            except Exception as e:
                raise RuntimeError(f"Error when fitting: {e}")
            res = {}
            for name_scorer, scorer in self.custom_scorer:
                try:
                    # pipe[-1] because (clustering) model at the end
                    score_val = scorer(X, pipe[-1].labels_)
                except Exception as e:
                    print(f"Error when calculating {name_scorer} on {name}")
                    print(e)
                    score_val = np.nan
                res[name_scorer] = score_val

            # store fitted models
            self.fitted_models_[name] = pipe

            # add metrics to the lists

            # nan mean are used, because when folding,
            # a label can be missing
            # in a particular fold, thus creating nan values
            for metric_name in self.custom_scorer.keys():
                print_metric_name = METRIC_NAMES.get(metric_name,
                                                     metric_name)
                (metrics_dic[print_metric_name]
                 .append(np.nanmean(res[metric_name])))
            metrics_dic["Model"].append(name)  # add name
            # add running time
            metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                metrics_dic[metric_name] = [-x for x in
                                            metrics_dic[metric_name]]
        df_models = pd.DataFrame.from_dict(metrics_dic)
        df_models = (df_models
                     .sort_values(by=self.sort_result_by,
                                  ascending=self.ascending)
                     .set_index("Model"))
        self.df_models = df_models

        if self.save_results:  # save results
            df_models.to_csv(opj.path.join(self.path_results,
                                           f"vulpes_results_{self.task}.csv"))
        return df_models
