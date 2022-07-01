from vulpes.automl import Clustering

import numpy as np


def test_clustering_1() -> None:
    # clustering
    X = np.array([[np.random.randint(-20, 20) * np.random.random(),
                   np.random.randint(-10, 10) * np.random.random()]
                 for _ in range(50)])
    clustering = Clustering()
    _ = clustering.fit(X)
