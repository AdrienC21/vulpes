#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""classifiers.py: Class Classifiers
to test many classification models
"""

from .corevulpes import CoreVulpes
from ..utils.utils import (
    CUSTOM_SCORER_CLF,
    METRIC_NAMES,
    METRICS_TO_REVERSE,
    create_model_2,
)

import warnings
import numbers
from time import perf_counter
from typing import List, Dict, Any, Union, Tuple
from collections import defaultdict
from collections.abc import Iterable
from os.path import join as opj

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    RepeatedKFold,
    _validation,
    train_test_split,
)

# define type Array_like
Array_like = Union[List, pd.DataFrame, pd.Series, np.ndarray, Any]


class Classifiers(CoreVulpes):
    """
    Object to train many classifiers.
    All the parameters are optionals and can be modified.

    Args:
        models_to_try (Union[str, List[Tuple[str, Any]]], optional):
            List of models to try. It can be either a string that corresponds
            to a predefined list of models ("all", ...) or it can be a list
            of tuple (name of a model, class of a model)
            (e.g. ("RandomForestClassifier",
                   sklearn.ensemble.RandomForestClassifier)).
            Defaults to "all" (train all the available classification
            algorithms).
        custom_scorer (Dict[str, Any], optional): metrics to calculate after
            fitting a model. Dictionary with pairs name:scorer where the
            scorer is created using the function make_scorer from sklearn.
            Defaults to CUSTOM_SCORER_CLF.
        preprocessing (Union[Pipeline, str], optional): preprocessing
            pipeline to use. It can be None (no preprocessing), a
            predefined preprocessing pipeline ("default", ...) or
            a Pipeline object from sklearn. Defaults to "default":
            it applies a OneHotEncoder to "category" and object features,
            and it applies a SimpleImputer (median strategy) and a
            StandardScaler to numerical features.
        use_cross_validation (bool, optional): whether or not we apply
            a cross-validation. Defaults to True.
        cv (Any, optional): cross-validation object. It can be a predefined
            cross-validation setting ("default", "timeseries", ...), an
            iterable object, a cross-validation object from sklearn, etc.
            Defaults to "default": it applies a StratifiedShuffleSplit
            if a groups object is given when applying the fit method,
            otherwise, it uses a RepeatedKFold. In both cases, n_splits
            is set to 5.
        test_size (float, optional): test of the size set when splitting.
            Defaults to 0.2.
        shuffle (bool, optional): whether or not the algorithm shuffle the
            sample when splitting the given dataset. Defaults to False.
        sort_result_by (str, optional): on which metric do you want to
            sort the final dataframe. Defaults to "Balanced Accuracy".
        ascending (bool, optional): sort the final dataframe in
            ascending order?. Defaults to False.
        save_results (bool, optional): if True, save the results in a csv
            file. Defaults to False.
        path_results (str, optional): path to use when saving the results.
            Defaults to "".
        additional_model_params (Dict[str, Any], optional): dictionary
            that contains parameters to be applied to each element of the
            pipeline. E.g. {"n_estimators": 100}, apply to all the
            preprocessing tasks and/or models that have the parameter
            n_estimators with the parameter n_estimators. Defaults to {}.
        random_state (int, optional): random state variable. Is applied
            to every model and elements of the pipeline. Defaults to 42.
        verbose (int, optional): if greater than 1, print the warnings.
            Defaults to 0.
    Examples:

        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from vulpes.automl import Classifiers

        >>> dataset = load_iris()
        >>> X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
        >>> y = dataset["target"]

        >>> classifiers = Classifiers()
        >>> df_models = classifiers.fit(X, y)
        >>> df_models
        | Model                       | Balanced Accuracy | Accuracy | ...
        |-----------------------------|-------------------|----------|------
        Model									
        LinearDiscriminantAnalysis	    0.977625	        0.977333   ...
        MLPClassifier	                0.973753	        0.973333   ...
        QuadraticDiscriminantAnalysis	0.973219	        0.973333   ...
        KNeighborsClassifier	        0.971702	        0.969333   ...
        ...                             ...                 ...        ...
    """
    def __init__(
        self,
        *,
        models_to_try: Union[str, List[Tuple[str, Any]]] = "all",
        custom_scorer: Dict[str, Any] = CUSTOM_SCORER_CLF,
        preprocessing: Union[Pipeline, str] = "default",
        use_cross_validation: bool = True,
        cv: Any = "default",
        test_size: float = 0.2,
        shuffle: bool = False,
        sort_result_by: str = "Balanced Accuracy",
        ascending: bool = False,
        save_results: bool = False,
        path_results: str = "",
        additional_model_params: Dict[str, Any] = {},
        random_state: int = 42,
        verbose: int = 0,
    ):
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
        self.additional_model_params = additional_model_params
        self.random_state = random_state
        self.verbose = verbose

        if not (self.verbose):
            warnings.filterwarnings("ignore")

    def fit(
        self,
        X: Array_like,
        y: Array_like,
        *,
        sample_weight: Array_like = None,
        groups: Array_like = None,
    ) -> pd.DataFrame:
        """
        Fit many models

        Args:
            X (Array_like): Input dataset
            y (Array_like): Target variable
            sample_weight (Array_like, optional): sample weights
                to apply when fitting. Defaults to None.
            groups (Array_like, optional): groups to use during
                cross validation to stratify. Defaults to None.

        Raises:
            ValueError: cross validation wrong type, or failed
            RuntimeError: Error when fitting a model

        Returns:
            pd.DataFrame: dataframe with the goodness-of-fit metrics
                evaluated for each model.

        Examples:

        """
        # Convert X to dataframe
        # (some preprocessing task, model, etc require this format)
        if not (isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)

        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        # Fit and calculate metrics for all the models
        for name, model in tqdm(self.models_to_try):
            print("\n" + name)
            top = perf_counter()

            # extend the class with a custom predict proba function
            # based on a decision function or pairwise distances
            if not (hasattr(model, "predict_proba")):
                model = create_model_2(model)
            else:
                model = model()
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
            for pipe_name, pipe_elt in pipe.steps:
                pipe_elt_available_params = pipe_elt.get_params().keys()
                if "random_state" in pipe_elt_available_params:
                    rd_state = f"{pipe_name}__random_state"
                    model_params[rd_state] = self.random_state
                if "normalize" in pipe_elt_available_params:
                    model_params[f"{pipe_name}__normalize"] = False
                if "n_jobs" in pipe_elt_available_params:
                    model_params[f"{pipe_name}__n_jobs"] = -1
                if "probability" in pipe_elt_available_params:
                    model_params[f"{pipe_name}__probability"] = True
                for (
                    add_param_name,
                    add_param_val,
                ) in self.additional_model_params.items():
                    if add_param_name in pipe_elt_available_params:
                        model_params[
                            f"{pipe_name}__{add_param_name}"
                        ] = add_param_val
            # change the loss to allow multiclass and predict proba
            if name == "SGDClassifier":
                model_params[f"{model_name}__loss"] = "log"
            if model_params != {}:
                pipe.set_params(**model_params)

            # if sample_weight is a parameter of the fit function, update
            # fit parameters
            fit_params = {}
            for pipe_name, pipe_elt in pipe.steps:
                if hasattr(pipe_elt, "fit") and (
                    "sample_weight" in pipe_elt.fit.__code__.co_varnames
                ):
                    fit_params[f"{pipe_name}__sample_weight"] = sample_weight

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
                        scoring=self.custom_scorer,
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
                        scoring=self.custom_scorer,
                    )
                except Exception as e:
                    raise RuntimeError(str(e))

                # store fitted models
                self.fitted_models_[name] = cv_model["estimator"]

                # add metrics to the lists

                # nan mean are used, because when folding,
                # a label can be missing
                # in a particular fold, thus creating nan values
                for metric_name in self.custom_scorer.keys():
                    print_metric_name = METRIC_NAMES.get(metric_name,
                                                         metric_name)
                    (
                        metrics_dic[print_metric_name].append(
                            np.nanmean(cv_model[f"test_{metric_name}"])
                        )
                    )
                metrics_dic["Model"].append(name)  # add name
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
                        pipe, X_test, y_test, self.custom_scorer,
                        error_score="raise"
                    )
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
                    (
                        metrics_dic[print_metric_name].append(
                            np.nanmean(res[metric_name])
                        )
                    )
                metrics_dic["Model"].append(name)  # add name
                # add running time
                metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                reverse_metric = [-x for x in metrics_dic[metric_name]]
                metrics_dic[metric_name] = reverse_metric
        # create a dataframe with the results
        df_models = pd.DataFrame.from_dict(metrics_dic)
        df_models = df_models.sort_values(
            by=self.sort_result_by, ascending=self.ascending
        ).set_index("Model")
        self.df_models = df_models

        if self.save_results:  # save results
            df_models.to_csv(
                opj.path.join(self.path_results,
                              f"vulpes_results_{self.task}.csv")
            )
        return df_models
