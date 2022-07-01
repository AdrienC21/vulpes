#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""regressions.py: Class Regressions
to test many regression models

@Author: Adrien Carrel
"""

from .corevulpes import CoreVulpes
from ..utils.utils import CUSTOM_SCORER_REG, METRIC_NAMES, \
    METRICS_TO_REVERSE, r2_score_adj

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
from sklearn.model_selection import cross_validate, RepeatedKFold, \
    _validation, train_test_split
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore")
# define type Array_like
Array_like = Union[List, pd.DataFrame, pd.Series, np.ndarray, Any]


class Regressions(CoreVulpes):
    def __init__(self, *,
                 models_to_try: Union[str, List[Tuple[str, Any]]] = "all",
                 custom_scorer: Dict[str, Any] = CUSTOM_SCORER_REG,
                 preprocessing: Union[Pipeline, str] = "default",
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
        # Convert X to dataframe
        # (some preprocessing task, model, etc require this format)
        if not(isinstance(X, pd.DataFrame)):
            X = pd.DataFrame(X)

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
            print("\n" + name)
            top = perf_counter()

            model = model()
            model_name = name.lower()
            # Create the pipeline
            pipe = self.create_pipeline(model_name, model)

            # When available:
            # - maximize n_jobs
            # - fix a random_state
            # - set normalize to False
            # (normalization should be handled by the preprocessing part)
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
