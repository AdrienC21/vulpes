#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""classifiers.py: Class Classifiers
to test many classification models

@Author: Adrien Carrel
"""

from .corevulpes import CoreVulpes
from ..utils.utils import CUSTOM_SCORER_CLF, METRIC_NAMES, \
    METRICS_TO_REVERSE, create_model_2

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
from sklearn.model_selection import cross_validate, RepeatedKFold, \
    _validation, train_test_split

# define type Array_like
Array_like = Union[List, pd.DataFrame, pd.Series, np.ndarray, Any]


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

    def fit(self, X: Array_like, y: Array_like, *,
            sample_weight: Array_like = None,
            groups: Array_like = None) -> pd.DataFrame:
        """Fit many models

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
        # dictionary to store calculated values, model info, etc for each model
        metrics_dic = defaultdict(list)

        # Fit and calculate metrics for all the models
        for name, model in tqdm(self.models_to_try):
            print("\n" + name)
            top = perf_counter()

            # extend the class with a custom predict proba function
            # based on a decision function or pairwise distances
            if not(hasattr(model, "predict_proba")):
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
                    model_params[f"{pipe_name}"
                                 "__random_state"] = self.random_state
                if "normalize" in pipe_elt_available_params:
                    model_params[f"{pipe_name}__normalize"] = False
                if "n_jobs" in pipe_elt_available_params:
                    model_params[f"{pipe_name}__n_jobs"] = -1
                if "probability" in pipe_elt_available_params:
                    model_params[f"{pipe_name}__probability"] = True
            # change the loss to allow multiclass and predict proba
            if name == "SGDClassifier":
                model_params[f"{model_name}__loss"] = "log"
            if model_params != {}:
                pipe.set_params(**model_params)

            # if sample_weight is a parameter of the fit function, update
            # fit parameters
            fit_params = {}
            for pipe_name, pipe_elt in pipe.steps:
                if (hasattr(pipe_elt, "fit") and
                   ("sample_weight" in pipe_elt.fit.__code__.co_varnames)):
                    fit_params[f"{pipe_name}__sample_weight"] = sample_weight

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
