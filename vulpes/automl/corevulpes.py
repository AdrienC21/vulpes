#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""corevulpes.py: Parent class that contains
common methods shared between children classes

@Author: Adrien Carrel
"""

from ..utils.utils import (
    CLASSIFIERS,
    REGRESSIONS,
    CLUSTERING,
    METRIC_NAMES,
    METRICS_TO_REVERSE,
    create_model_2,
)

import warnings
import numbers
import operator
from functools import reduce
from time import perf_counter
from typing import List, Dict, Any, Union, Tuple
from collections import defaultdict
from collections.abc import Iterable
from abc import ABC

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    RepeatedKFold,
    StratifiedShuffleSplit,
    _validation,
    train_test_split,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError
from sklearn.metrics._scorer import _ProbaScorer, _PredictScorer

warnings.filterwarnings("ignore")
# define type Array_like
Array_like = Union[List, pd.DataFrame, pd.Series, np.ndarray, Any]


class CoreVulpes(ABC):
    """
    Parent class with shared methods for the classes
    Classifiers, Regressions and Clustering
    """

    def __init__(self):
        # store the metrics dataframe after fitting
        self.df_models = None  # all the models
        self.df_best_model = None  # the best one with a voting clf/reg
        # store the fitted models
        self.fitted_models_ = {}  # all the models
        self.best_model_ = None  # the best one with a voting clf/reg

    def missing_data(self, X: Array_like) -> pd.DataFrame:
        """
        Evaluate the absolute count and the percentage of
        missing data in a particular dataset

        Args:
            X (Array_like): Dataset

        Returns:
            pd.DataFrame: Absolute count and percentage of missing
            data in X

        Examples:

        """
        if not (isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)  # locally modify X
        total_missing = X.isnull().sum().sort_values(ascending=False)
        percent_missing = (X.isnull().sum() / X.isnull().count()).sort_values(
            ascending=False
        ) * 100
        missing = pd.concat(
            [total_missing, percent_missing],
            axis=1,
            keys=["Total Missing", "Percentage (%)"],
        )
        return missing

    def predefined_preprocessing(self) -> Pipeline:
        """
        Either return a predefined preprocessing pipeline
        if self.preprocessing is a string. Otherwise, check
        if self.preprocessing is in fact a Pipeline or a None
        object (in that case, no preprocessing applied).

        Raises:
            ValueError: Unknown string
            TypeError: self.preprocessing is not a string,
            not None, not a pipeline object

        Returns:
            Pipeline: None of preprocessing Pipeline to apply to each models

        Examples:

        """
        if isinstance(self.preprocessing, str):
            if self.preprocessing == "default":
                # Imputer + standard scaler for not categorical values
                numeric_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                # OneHotEncoder for categorical values
                categorical_transformer = OneHotEncoder(handle_unknown="ignore")

                preprocessing = ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            numeric_transformer,
                            selector(dtype_exclude=["category", object]),
                        ),
                        (
                            "cat",
                            categorical_transformer,
                            selector(dtype_include=["category", object]),
                        ),
                    ]
                )

                preprocessing_pipeline = Pipeline(
                    steps=[("preprocessing", preprocessing)]
                )
            # Insert predefined preprocessing pipelines here!
            else:
                raise ValueError(
                    "Unknown parameter: preprocessing."
                    "Please enter a valid preprocessing "
                    "(an already implemented one, "
                    ", a Pipeline object, or None)."
                )
        # if it's not a string, not a pipeline, or not None, raise error
        elif not (
            isinstance(self.preprocessing, Pipeline) or (self.preprocessing is None)
        ):
            raise TypeError(
                "Preprocessing must be a string, " "a Pipeline object, or None."
            )
        else:  # return the given preprocessing pipeline or None object
            preprocessing_pipeline = self.preprocessing
        return preprocessing_pipeline

    def create_pipeline(self, model_name: str, model: Any) -> Pipeline:
        """
        Create a pipeline by combining an optional preprocessing
        pipeline and the given model

        Args:
            model_name (str): name of the model (lowercase)
            model (Any): Model at the end of the pipeline

        Returns:
            Pipeline: Pipeline with a preprocessing task (if not set to None)
            and the given model

        Examples:

        """
        # Preprocessing
        preprocessing_pipeline = self.predefined_preprocessing()

        # Model
        model_pipeline = Pipeline(steps=[(model_name, model)])

        # Merge all the steps
        pipelines = [preprocessing_pipeline, model_pipeline]
        steps = reduce(operator.add, [p.steps for p in pipelines if not (p is None)])
        return Pipeline(steps=steps)

    def predefined_cv(self, X: Array_like = None, groups: Array_like = None) -> Any:
        """
        Convert a cross validation string (self.cv parameter)
        into a predefined cross validation object

        Args:
            X (Array_like, optional): if necessary, X is the dataset.
            Defaults to None.
            groups (Array_like, optional): if necessary, groups is
            an array-like object on which we stratify to create
            the different folds. Defaults to None.

        Raises:
            ValueError: raise an error if the string doesn't correspond
            to a predefined cross validation

        Returns:
            Any: Cross validation object

        Examples:

        """
        if self.cv == "default":
            # if groups, cross validation is a
            # stratified shuffle, else,
            # a repeatedKFold
            if groups is None:
                cv = RepeatedKFold(
                    n_splits=5, n_repeats=5, random_state=self.random_state
                )
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=5, test_size=self.test_size, random_state=self.random_state
                )
                cv = sss.split(X, groups)
        elif self.cv == "timeseries":
            cv = TimeSeriesSplit(n_splits=5)

        # Insert predefined cross validation here!
        else:
            raise ValueError(f"Unknown cross validation: {self.cv}")
        return cv

    def predefined_list_models(
        self, models_to_try: Union[str, List[Tuple[str, Any]]] = "all"
    ) -> List[Tuple[str, Any]]:
        """
        If models_to_try isn't a list of models but is a string,
        return the corresponding predefined list of models to test

        Args:
            models_to_try (Union[str, List[Tuple[str, Any]]],
                           optional): string of predefined list of models
            or list of tuple (name of model, model). Defaults to "all".

        Raises:
            ValueError: raise an error if models_to_try is a string that
            doesn't correspond to any predefined list of models

        Returns:
            List[Tuple[str, Any]]: list of tuple (name of the model, model)

        Examples:

        """
        if not (isinstance(models_to_try, str)):
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
                f"Unknown parameter models_to_try: {models_to_try}. "
                "Please enter a valid list of models (tuple like "
                '("XGBClassifier", xgboost.XGBClassifier)) or an '
                'existing predefined list of models("all", ...)'
            )

    def remove_proba_metrics(
        self, dic_scorer: Dict[str, Union[_ProbaScorer, _PredictScorer]]
    ) -> Dict[str, Union[_ProbaScorer, _PredictScorer]]:
        """
        Take a dictionnary of metrics to evaluate the goodness-of-fit of
        classifiers as an input. Return a new version of this
        dictionnary with only the metrics that don't need
        probabilities to be calculated (e.g. AUROC)

        Args:
            dic_scorer (Dict[str, Union[_ProbaScorer, _PredictScorer]]):
            dictionnary of metrics

        Returns:
            Dict[str, Union[_ProbaScorer, _PredictScorer]]: filtered
            dictionnary of metrics

        Examples:

        """
        new_dic = {}
        for scorer_name, scorer in dic_scorer.items():
            if isinstance(scorer, _PredictScorer):  # no predict proba
                new_dic[scorer_name] = scorer
        return new_dic

    def build_best_models(
        self,
        X: Array_like,
        y: Array_like,
        *,
        sample_weight: Array_like = None,
        groups: Array_like = None,
        nb_models: int = 5,
        sort_result_by: str = None,
        ascending: bool = None,
        voting: str = "hard",
        weights: Array_like = None,
    ) -> pd.DataFrame:
        """
        When many models have been fitted, create an aggregated model
        using a voting system by selecting the nb_models best models based on
        the metric sort_result_by.

        Args:
            X (Array_like): dataset to fit the 'best model'
            y (Array_like): response/outcome variable
            sample_weight (Array_like, optional): sample weight.
                Defaults to None.
            groups (Array_like, optional): groups to stratify during
                cross validation. Defaults to None.
            nb_models (int, optional): number of models to select when
                creating the aggregated model. Defaults to 5.
            sort_result_by (str, optional): metrics to evaluate the best
                models that will be selected among the ones that we trained.
                Defaults to None.
            ascending (bool, optional): if ascending=True, the lower the
                metric sort_result_by is, the better the model is.
                Defaults to None.
            voting (str, optional): "hard" or "soft". If "soft", use
                the predicted probabilities of the different estimators
                to make a prediction. Defaults to "hard".
            weights (Array_like, optional): attribute different weights
                to each estimators. Defaults to None, which is equal to
                equal weights.

        Raises:
            ValueError: Voting Clustering not available
            NotFittedError: Fit models before building a 'best model'
            ValueError: less fitted models that the parameter nb_models
            NotImplementedError: Voting Clustering not available
            NotImplementedError: Sample weight not available
            ValueError: Wrong type of cross validation
            RuntimeError: Error when fitting an estimator

        Returns:
            pd.DataFrame: Performance of the aggregated model
                on different metrics

        Examples:

        """
        if self.task == "clustering":
            raise ValueError("Can't create a Voting Clustering algorithm.")
        if self.df_models is None:
            raise NotFittedError(
                "Please fit the models first by calling the method .fit "
                "before building an ensemble model"
            )
        if nb_models > len(self.df_models):
            raise ValueError(
                f"Not enough trained models ({len(self.df_models)}) "
                f"to select the best {nb_models} ones"
            )

        top = perf_counter()  # start to measure fitting time

        # Convert X to dataframe
        # (some preprocessing task, model, etc require this format)
        if not (isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)

        # if undefined, take default values for sort_result_by and ascending
        # to select the best models
        if sort_result_by is None:
            sort_result_by = self.sort_result_by
        if ascending is None:
            ascending = self.ascending
        # sort the models based on the given metric
        sorted_df_models = self.df_models.sort_values(
            by=sort_result_by, ascending=ascending
        )
        # name of the best models
        best_models_names = list(sorted_df_models.index)[:nb_models]
        # dictionnary: model name -> corresponding class
        dict_models = dict(self.predefined_list_models(self.models_to_try))
        # list of tuple (model name, instance of model)

        # create the list of estimators
        if self.task == "classification":  # check predict_proba method
            estimators = []
            for b in best_models_names:
                if not (hasattr(dict_models[b], "predict_proba")):
                    estimator = create_model_2(dict_models[b])
                else:
                    estimator = dict_models[b]()
                estimators.append((b.lower(), estimator))
        else:
            estimators = [(b.lower(), dict_models[b]()) for b in best_models_names]

        # define the voting model
        if self.task == "classification":
            voting = VotingClassifier(
                estimators, voting=voting, weights=weights, n_jobs=-1
            )
        elif self.task == "regression":
            voting = VotingRegressor(estimators, weights=weights, n_jobs=-1)
        else:
            raise NotImplementedError("Unknown task: {self.task}")
        voting_name = voting.__class__.__name__.lower()
        # Add preprocessing, create a pipeline
        pipe = self.create_pipeline(voting_name, voting)
        # adjust some hyperparameters when available for each model inside the
        # voting model and the pipeline
        model_params = {}
        for pipe_name, pipe_elt in pipe.steps:
            pipe_elt_available_params = pipe_elt.get_params().keys()
            if "random_state" in pipe_elt_available_params:
                model_params[f"{pipe_name}" "__random_state"] = self.random_state
            if "normalize" in pipe_elt_available_params:
                model_params[f"{pipe_name}__normalize"] = False
            if "n_jobs" in pipe_elt_available_params:
                model_params[f"{pipe_name}__n_jobs"] = -1
            for add_param_name, add_param_val in self.additional_model_params.items():
                if add_param_name in pipe_elt_available_params:
                    model_params[f"{pipe_name}" f"__{add_param_name}"] = add_param_val

        for (model_name, model_instance) in estimators:
            model_available_params = model_instance.get_params().keys()
            if "random_state" in model_available_params:
                model_params[
                    f"{voting_name}__{model_name}__" "random_state"
                ] = self.random_state
            if "n_jobs" in model_available_params:
                model_params[f"{voting_name}__{model_name}__n_jobs"] = -1
            if "probability" in model_available_params:
                model_params[f"{voting_name}__{model_name}__" "probability"] = True
            if "normalize" in model_available_params:
                model_params[f"{voting_name}__{model_name}__" "normalize"] = False
            for add_param_name, add_param_val in self.additional_model_params.items():
                if add_param_name in model_available_params:
                    model_params[
                        f"{voting_name}__{pipe_name}" f"__{add_param_name}"
                    ] = add_param_val
            # change the loss to allow multiclass and predict proba
            if model_name == "SGDClassifier".lower():
                model_params[f"{voting_name}__{model_name}__loss"] = "log"
        if model_params != {}:
            pipe.set_params(**model_params)

        ######
        # Voting Classifier / Regressor don't really support sample weight
        # all of the models need a sample weight :/
        if not (sample_weight is None):
            raise NotImplementedError(
                "Sample weight not (yet) implemented when fitting a" " 'best model'."
            )
        fit_params = {}
        ######

        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        # if it's a voting classifier with "hard" voting, as it
        # doesn't (yet) allow the predict proba method, we need to
        # remove some metrics when evaluating the performance
        # of the model
        if (self.task == "classification") and (
            pipe[-1].get_params()["voting"] == "hard"
        ):
            scoring = self.remove_proba_metrics(self.custom_scorer)
        else:
            scoring = self.custom_scorer

        if self.use_cross_validation:
            if not (
                hasattr(self.cv, "split")
                or isinstance(self.cv, numbers.Integral)
                or isinstance(self.cv, Iterable)
                or isinstance(self.cv, str)
            ):
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
                    pipe,
                    X,
                    y,
                    cv=cv,
                    return_estimator=True,
                    n_jobs=-1,
                    fit_params=fit_params,
                    scoring=scoring,
                )
            except ValueError as e:
                print(e)
                print("Cross validation failed. If groups provided, ")
                print(
                    "maybe can't stratify because some groups "
                    "represented in the dataset don't contains enough"
                    " samples."
                )
                print("Using RepeatedKFold instead.")
                cv = RepeatedKFold(
                    n_splits=5, n_repeats=5, random_state=self.random_state
                )
                cv_model = cross_validate(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    return_estimator=True,
                    n_jobs=-1,
                    fit_params=fit_params,
                    scoring=scoring,
                )
            except Exception as e:
                raise RuntimeError(str(e))

            # add metrics to the lists

            # nan mean are used, because when folding,
            # a label can be missing
            # in a particular fold, thus creating nan values
            for metric_name in scoring.keys():
                print_metric_name = METRIC_NAMES.get(metric_name, metric_name)
                (
                    metrics_dic[print_metric_name].append(
                        np.nanmean(cv_model[f"test_{metric_name}"])
                    )
                )

            metrics_dic["Model"].append(f"Voting ({nb_models}-best)")
            # add running time
            metrics_dic["Running time"].append(perf_counter() - top)
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    shuffle=self.shuffle,
                    stratify=groups,
                    random_state=self.random_state,
                )
                pipe.fit(X_train, y_train, **fit_params)
                res = _validation._score(
                    pipe, X_test, y_test, scoring, error_score="raise"
                )
            except Exception as e:
                raise RuntimeError(f"Error when fitting: {e}")

            # add metrics to the lists

            # nan mean are used, because when folding,
            # a label can be missing
            # in a particular fold, thus creating nan values
            for metric_name in scoring.keys():
                print_metric_name = METRIC_NAMES.get(metric_name, metric_name)
                (metrics_dic[print_metric_name].append(np.nanmean(res[metric_name])))
            metrics_dic["Model"].append(f"Voting ({nb_models}-best)")
            # add running time
            metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                metrics_dic[metric_name] = [-x for x in metrics_dic[metric_name]]

        # Metric for the fitted "best model"
        df_best_model = pd.DataFrame.from_dict(metrics_dic)
        df_best_model = df_best_model.set_index("Model")
        self.best_model_ = pipe
        self.df_best_model = df_best_model

        return df_best_model

    def get_fitted_models(self) -> Dict[str, Union[Pipeline, List[Pipeline]]]:
        """
        Get a dictionnary with the fitted models

        Raises:
            NotFittedError: Models have not been fitted yet

        Returns:
            Dict[str, Union[Pipeline, List[Pipeline]]]: Dictionnary
            with, for all models, either the fitted model, or all
            the fitted models (during cross validation).

        Examples:

        """

        if self.fitted_models_ is None:
            raise NotFittedError(
                "Fit some models before retrieving them by calling " "the method .fit"
            )
        return self.fitted_models_

    def get_best_model(self) -> Pipeline:
        """
        Get the model created and fitted by the
        method build_best_models

        Raises:
            NotFittedError: Models not trained
            NotFittedError: Best model not calculated

        Returns:
            Pipeline: 'Best model' using multiple fitted models

        Examples:

        """
        if self.best_model_ is None:
            if self.fitted_models_ is None:
                raise NotFittedError(
                    "Fit some models before retrieving them by calling "
                    "the method .fit"
                )
            raise NotFittedError(
                "Many models have been fitted. But the 'best model' "
                "hasn't been fitted yet. Please call the method "
                ".build_best_models before retrieving it."
            )
        return self.best_model_

    def predict(
        self, X: Array_like, *, dataframe_format: bool = True
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Evaluate all the fitted models on the dataset X

        Args:
            X (Array_like): Array-like object on which we'll make prediction(s)
            dataframe_format (bool, optional): if True, then the result
            is a dataframe with all the predictions for all the models.
            Defaults to True.

        Returns:
            Union[pd.DataFrame, Dict[str, np.ndarray]]: Dictionnary or
            dataframe with the predictions

        Examples:

        """
        fitted_models = self.get_fitted_models()
        res = {}
        for name, fitted_model in fitted_models.items():
            if isinstance(fitted_model, List) or isinstance(fitted_model, np.ndarray):
                print(
                    "Cross validation used to fit. Select the model"
                    " fitted on the first fold."
                )
                fitted_model = fitted_model[0]
            res[name] = fitted_model.predict(X)
        if not (dataframe_format):
            return res
        return pd.DataFrame.from_dict(res)

    def predict_proba(
        self,
        X: Array_like,
    ) -> Dict[str, np.ndarray]:
        """
        Based on the fitted models, make many probability
        predictions on the dataset X

        Args:
            X (Array_like): Dataset

        Returns:
            Dict[str, np.ndarray]: dictionnary with, for each model,
            an array of the corresponding predicted probabilities

        Examples:

        """
        if self.task == "clustering":
            return NotImplementedError(
                "predict_proba method not (yet) implemented for "
                "clustering algorithms"
            )
        fitted_models = self.get_fitted_models()
        res = {}
        for name, fitted_model in fitted_models.items():
            if isinstance(fitted_model, List) or isinstance(fitted_model, np.ndarray):
                print(
                    "Cross validation used to fit. Select the model"
                    " fitted on the first fold."
                )
                fitted_model = fitted_model[0]
            res[name] = fitted_model.predict_proba(X)
        return res

    def predict_best(self, X: Array_like) -> np.ndarray:
        """
        Evaluate the fitted 'best model' on the array-like X

        Args:
            X (Array_like): dataset

        Returns:
            np.ndarray: array of predictions

        Examples:

        """
        best_model = self.get_best_model()
        if isinstance(best_model, List) or isinstance(best_model, np.ndarray):
            print(
                "Cross validation used to fit. Select the model"
                " fitted on the first fold."
            )
            best_model = best_model[0]
        return best_model.predict(X)

    def predict_proba_best(self, X: Array_like) -> np.ndarray:
        """
        Evaluate the fitted 'best model' on the array-like X
        and return probabilities

        Args:
            X (Array_like): dataset

        Returns:
            np.ndarray: predicted probabilities

        Examples:

        """
        best_model = self.get_best_model()
        if isinstance(best_model, List) or isinstance(best_model, np.ndarray):
            print(
                "Cross validation used to fit. Select the model"
                " fitted on the first fold."
            )
            best_model = best_model[0]
        return best_model.predict_proba(X)
