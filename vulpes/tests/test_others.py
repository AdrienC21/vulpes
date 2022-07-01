from vulpes.automl import Classifiers

import numpy as np


def test_others_1() -> None:
    X = np.array([[np.random.randint(-20, 20) * np.random.random(),
                   np.random.randint(-10, 10) * np.random.random()]
                 for _ in range(50)])
    X[0][1] = np.nan
    X[1][0] = None
    classifiers = Classifiers()
    _ = classifiers.missing_data(X)
