from vulpes.automl import Classifiers

import numpy as np


def test_classifiers_1() -> None:
    # binary output + hard voting
    X = np.array([[np.random.randint(-20, 20) * np.random.random(),
                   np.random.randint(-10, 10) * np.random.random()]
                 for _ in range(50)])
    y = np.array([np.random.randint(0, 2) for _ in range(50)])
    classifiers = Classifiers()
    _ = classifiers.fit(X, y)
    _ = classifiers.build_best_models(X, y, voting="hard")


def test_classifiers_2() -> None:
    # 3-class + soft voting + sample weight + groups
    X = np.array([[np.random.randint(-20, 20) * np.random.random(),
                   np.random.randint(-10, 10) * np.random.random()]
                 for _ in range(50)])
    y = np.array([np.random.randint(0, 3) for _ in range(50)])
    sample_weight = np.array([np.random.random() for _ in range(50)])
    groups = np.array([np.random.randint(0, 2) for _ in range(50)])
    classifiers = Classifiers()
    _ = classifiers.fit(X, y, sample_weight=sample_weight,
                        groups=groups)
    _ = classifiers.build_best_models(X, y, voting="hard")


def test_classifiers_3() -> None:
    # default preprocessing, no cross validation
    # + test get_ functions to retrieve models and predict functions
    # on fitted models
    X = np.array([[np.random.randint(-20, 20) * np.random.random(),
                   np.random.randint(-10, 10) * np.random.random()]
                 for _ in range(50)])
    y = np.array([np.random.randint(0, 2) for _ in range(50)])
    classifiers = Classifiers(preprocessing="default",
                              use_cross_validation=False)
    _ = classifiers.fit(X, y)
    _ = classifiers.build_best_models(X, y, voting="hard")

    _ = classifiers.get_fitted_models()
    _ = classifiers.get_best_model()
    _ = classifiers.predict(X)
    _ = classifiers.predict_proba(X)
    _ = classifiers.predict_best(X)
    _ = classifiers.predict_proba_best(X)
