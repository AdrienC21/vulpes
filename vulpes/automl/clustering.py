#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""clustering.py: Class Clustering
to test many clustering algorithms
"""

from .corevulpes import CoreVulpes
from ..utils.utils import CUSTOM_SCORER_CLT, METRIC_NAMES, METRICS_TO_REVERSE

import warnings
from time import perf_counter
from typing import List, Dict, Any, Union, Tuple
from collections import defaultdict
from os.path import join as opj

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline

# define type Array_like
Array_like = Union[List, pd.DataFrame, pd.Series, np.ndarray, Any]


class Clustering(CoreVulpes):
    """
    Object to train many regressions.
    All the parameters are optionals and can be modified.

    Args:
        models_to_try (Union[str, List[Tuple[str, Any]]], optional):
            List of models to try. It can be either a string that corresponds
            to a predefined list of models ("all", ...) or it can be a list
            of tuple (name of a model, class of a model)
            (e.g. ("KMeans",
                   sklearn.cluster.KMeans)).
            Defaults to "all" (train all the available clustering
            algorithms).
        custom_scorer (Dict[str, Any], optional): metrics to calculate after
            fitting a model. Dictionary with pairs name:scorer where the
            scorer is created using the function make_scorer from sklearn.
            Defaults to CUSTOM_SCORER_CLT.
        preprocessing (Union[Pipeline, str], optional): preprocessing
            pipeline to use. It can be None (no preprocessing), a
            predefined preprocessing pipeline ("default", ...) or
            a Pipeline object from sklearn. Defaults to "default":
            it applies a OneHotEncoder to "category" and object features,
            and it applies a SimpleImputer (median strategy) and a
            StandardScaler to numerical features.
        sort_result_by (str, optional): on which metric do you want to
            sort the final dataframe. Defaults to "Davies–Bouldin Index".
        ascending (bool, optional): sort the final dataframe in
            ascending order?. Defaults to True.
        save_results (bool, optional): if True, save the results in a csv
            file. Defaults to False.
        path_results (str, optional): path to use when saving the results.
            Defaults to "".
        additional_model_params (Dict[str, Any], optional): dictionary
            that contains parameters to be applied to each element of the
            pipeline. E.g. {"n_estimators": 100}, apply to all the
            preprocessing tasks and/or models that have the parameter
            n_estimators with the parameter n_estimators.
            Defaults to {"nb_clusters": 3, "min_samples": 5, "eps": 0.5}.
        random_state (int, optional): random state variable. Is applied
            to every model and elements of the pipeline. Defaults to 42.
        verbose (int, optional): if greater than 1, print the warnings.
            Defaults to 0.
    """
    def __init__(
        self,
        *,
        models_to_try: Union[str, List[Tuple[str, Any]]] = "all",
        custom_scorer: Dict[str, Any] = CUSTOM_SCORER_CLT,
        preprocessing: Union[Pipeline, str] = "default",
        sort_result_by: str = "Davies–Bouldin Index",
        ascending: bool = True,
        save_results: bool = False,
        path_results: str = "",
        additional_model_params: Dict[str, Any] = {"nb_clusters": 3,
                                                   "min_samples": 5,
                                                   "eps": 0.5},
        random_state: int = None,
        verbose: int = 0,
    ):
        super().__init__()
        self.task = "clustering"
        self.models_to_try = self.predefined_list_models(models_to_try)
        self.custom_scorer = custom_scorer
        self.preprocessing = preprocessing
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
        *,
        sample_weight: Array_like = None
    ) -> pd.DataFrame:
        """
        Fit many clustering algorithms

        Args:
            X (Array_like): Input dataset
            sample_weight (Array_like, optional): sample weights
                to apply when fitting. Defaults to None.

        Raises:
            RuntimeError: Error when fitting a model

        Returns:
            pd.DataFrame: dataframe with the goodness-of-fit metrics
                evaluated for each clustering algorithm.

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

            model = model()
            model_name = name.lower()
            # Create the pipeline
            pipe = self.create_pipeline(model_name, model)

            # When available:
            # - maximize n_jobs
            # - fix a random_state
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
                for (
                    add_param_name,
                    add_param_val,
                ) in self.additional_model_params.items():
                    if add_param_name in pipe_elt_available_params:
                        model_params[
                            f"{pipe_name}__{add_param_name}"
                        ] = add_param_val
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

            try:
                pipe.fit(X, **fit_params)
            except Exception as e:
                raise RuntimeError(f"Error when fitting: {e}")
            res = {}
            for name_scorer, scorer in self.custom_scorer.items():
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
                # print_metric_name (pmn)
                pmn = METRIC_NAMES.get(metric_name, metric_name)
                (metrics_dic[pmn].append(np.nanmean(res[metric_name])))
            metrics_dic["Model"].append(name)  # add name
            # add running time
            metrics_dic["Running time"].append(perf_counter() - top)

        # reverse these metrics (because lower is better)
        # ex: rmse, mae, mape
        for metric_name in METRICS_TO_REVERSE:
            if metric_name in metrics_dic:
                reverse_metric = [-x for x in metrics_dic[metric_name]]
                metrics_dic[metric_name] = reverse_metric
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
