from vulpes.automl import Regressions

import numpy as np


def test_regressions_1() -> None:
    # regression with "timeseries" predefined cross validation
    X = np.array([[np.random.randint(-20, 20) * np.random.random(),
                   np.random.randint(-10, 10) * np.random.random()]
                 for _ in range(50)])
    y = np.array([np.random.randint(-5, 5) * np.random.random()
                  for _ in range(50)])
    regressions = Regressions(cv="timeseries")
    _ = regressions.fit(X, y)
    _ = regressions.build_best_models(X, y)
