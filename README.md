# Vulpes

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pypi version](https://img.shields.io/pypi/v/vulpes.svg)](https://pypi.python.org/pypi/vulpes)
[![Documentation Status](https://readthedocs.org/projects/vulpes/badge/?version=latest)](https://vulpes.readthedocs.io/en/latest/?badge=latest)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=AdrienC21.vulpes&right_color=%23FFA500)](https://github.com/AdrienC21/vulpes)
[![Downloads](https://static.pepy.tech/badge/vulpes)](https://pepy.tech/project/vulpes)

<img src="https://github.com/AdrienC21/vulpes/blob/main/logo_large.png?raw=true"  width=60% height=60%>

**Vulpes: Test many classification, regression models and clustering algorithms to see which one is most suitable for your dataset.**

Vulpes 🦊 is a Python package that allows you to test many models, whether you want to do classification, regression or clustering in your projects. It calculates many metrics for each model to compare them. It is highly customizable and it contains many features to save time building robust ML models.

If you like this project, please leave a star ⭐ on GitHub !

Alpha version.

Author & Maintainer: Adrien Carrel.

## Installation

Using pip:

```python
pip install vulpes
```

## Dependencies

vulpes requires:

- Python (>= 3.7)
- numpy (>= 1.22)
- pandas (>= 1.3.5)
- scikit-learn (>= 1.0.2)
- tqdm (>= 4.64.0)
- xgboost (>= 1.6.1)
- lightgbm (>= 3.3.2)

## Documentation

Link to the documentation: [https://vulpes.readthedocs.io/en/latest/](https://vulpes.readthedocs.io/en/latest/)

## Examples

General case, import one of the classes Classifiers, Regressions, Clustering from vulpes.automl, add some parameters to the object (optional), fit your dataset:

```python
from vulpes.automl import Classifiers
classifiers = Classifiers()
classifiers.fit(X, y)
```

More examples below and in notebooks in the folter **examples**.

### Classification

Fit many classification algorithms on the iris dataset from scikit-learn:

```python
import pandas as pd
from sklearn.datasets import load_iris
from vulpes.automl import Classifiers

dataset = load_iris()
X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
y = dataset["target"]

classifiers = Classifiers(preprocessing="default")
df_models = classifiers.fit(X, y)
df_models
```

Analysis of each model using different metrics and repeated cross-validation by K-fold:

|                          Model | Balanced Accuracy | Accuracy | Precision |   Recall | F1 Score |    AUROC |    AUPRC | Micro avg Precision | Running time |
|-------------------------------:|------------------:|---------:|----------:|---------:|---------:|---------:|---------:|--------------------:|-------------:|
|   LinearDiscriminantAnalysis   |          0.977625 | 0.977333 |  0.978024 | 0.977625 | 0.976933 | 0.998161 | 0.996891 |            0.996940 |     4.372556 |
|  QuadraticDiscriminantAnalysis |          0.973219 | 0.973333 |  0.975460 | 0.973219 | 0.973162 | 0.999063 | 0.997595 |            0.997634 |     4.470590 |
|      LogisticRegressionCV      |          0.961609 | 0.961333 |  0.964101 | 0.961609 | 0.960668 | 0.997218 | 0.993264 |            0.993375 |    12.895212 |
|               SVC              |          0.961287 | 0.960000 |  0.962045 | 0.961287 | 0.959960 | 0.996825 | 0.994421 |            0.994510 |     4.437862 |
|     RandomForestClassifier     |          0.957220 | 0.956000 |  0.959982 | 0.957220 | 0.955394 | 0.993473 | 0.990367 |            0.989958 |    10.645725 |
|           GaussianNB           |          0.957169 | 0.954667 |  0.956188 | 0.957169 | 0.954521 | 0.993825 | 0.990463 |            0.990619 |     4.345500 |
|      ExtraTreesClassifier      |          0.956438 | 0.956000 |  0.958665 | 0.956438 | 0.955157 | 0.995156 | 0.991795 |            0.991704 |    10.440453 |
|       LogisticRegression       |          0.956094 | 0.954667 |  0.957273 | 0.956094 | 0.954427 | 0.997726 | 0.994765 |            0.994848 |     5.691309 |
|   GradientBoostingClassifier   |          0.955871 | 0.953333 |  0.956984 | 0.955871 | 0.953364 | 0.983221 | 0.967145 |            0.971317 |     9.005045 |
|          XGBClassifier         |          0.952846 | 0.950667 |  0.952745 | 0.952846 | 0.950324 | 0.985892 | 0.969083 |            0.972853 |     4.802282 |
|        BaggingClassifier       |          0.952712 | 0.950667 |  0.955214 | 0.952712 | 0.950581 | 0.985295 | 0.982312 |            0.971742 |     8.354026 |
|      KNeighborsClassifier      |          0.952699 | 0.950667 |  0.951586 | 0.952699 | 0.950683 | 0.990842 | 0.986716 |            0.980262 |     6.960091 |
|       AdaBoostClassifier       |          0.950432 | 0.946667 |  0.949250 | 0.950432 | 0.947114 | 0.988202 | 0.981889 |            0.977999 |     8.127254 |
|         LGBMClassifier         |          0.950009 | 0.948000 |  0.950426 | 0.950009 | 0.947522 | 0.991721 | 0.985483 |            0.985704 |     5.063474 |
|         LabelSpreading         |          0.948757 | 0.945333 |  0.947960 | 0.948757 | 0.946091 | 0.988827 | 0.981177 |            0.981552 |     4.332253 |
| HistGradientBoostingClassifier |          0.948195 | 0.945333 |  0.949260 | 0.948195 | 0.945352 | 0.988212 | 0.976375 |            0.976866 |     7.706454 |
|        LabelPropagation        |          0.946091 | 0.944000 |  0.946373 | 0.946091 | 0.944250 | 0.990341 | 0.984098 |            0.984373 |     4.406253 |
|          MLPClassifier         |          0.944773 | 0.941333 |  0.945336 | 0.944773 | 0.942314 | 0.992075 | 0.985516 |            0.985762 |     7.662322 |
|     DecisionTreeClassifier     |          0.942681 | 0.941333 |  0.944493 | 0.942681 | 0.940183 | 0.957011 | 0.951111 |            0.908000 |     4.367503 |
|            LinearSVC           |          0.936713 | 0.936000 |  0.937548 | 0.936713 | 0.933929 | 0.989648 | 0.983251 |            0.983539 |     4.474272 |
|       ExtraTreeClassifier      |          0.933964 | 0.932000 |  0.934967 | 0.933964 | 0.931137 | 0.950473 | 0.943333 |            0.893289 |     4.336813 |
|          SGDClassifier         |          0.922581 | 0.918667 |  0.927593 | 0.922581 | 0.919651 | 0.981940 | 0.962839 |            0.963484 |     5.666082 |
|     CalibratedClassifierCV     |          0.894860 | 0.888000 |  0.896616 | 0.894860 | 0.887397 | 0.972231 | 0.957643 |            0.958332 |     5.699280 |
|           Perceptron           |          0.873581 | 0.865333 |  0.887799 | 0.873581 | 0.864172 | 0.976069 | 0.945789 |            0.946695 |     4.482433 |
|         NearestCentroid        |          0.854566 | 0.854667 |  0.854707 | 0.854566 | 0.849341 | 0.973214 | 0.963677 |            0.964257 |     5.783815 |
|         RidgeClassifier        |          0.843743 | 0.834667 |  0.848879 | 0.843743 | 0.831310 | 0.945148 | 0.920905 |            0.922219 |     4.415888 |
|        RidgeClassifierCV       |          0.841049 | 0.832000 |  0.846498 | 0.841049 | 0.828592 | 0.944421 | 0.919460 |            0.920816 |     4.484041 |
|           BernoulliNB          |          0.757425 | 0.758667 |  0.771867 | 0.757425 | 0.728847 | 0.883542 | 0.839397 |            0.823834 |     4.479535 |
|         DummyClassifier        |          0.333333 | 0.249333 |  0.083111 | 0.333333 | 0.132452 | 0.500000 | 0.379100 |            0.299444 |     4.396426 |
|                                |                   |          |           |          |          |          |          |                     |              |

Here, the "default" preprocessing pipeline has been used. It consists of SimpleImputer (median strategy) with a StandardScaler for the features and a OneHotEncoder for the categorical features.

### Regressions

Fit many regression algorithms:

```python
from sklearn.datasets import make_regression
from vulpes.automl import Regressions

X, y = make_regression(
          n_samples=100, n_features=4, random_state=42, noise=4.0,
          bias=100.0)

regressions = Regressions()
df_models = regressions.fit(X, y)
df_models
```

### Clustering

Fit many clustering algorithms on the iris dataset from scikit-learn:

```python
import pandas as pd
from sklearn.datasets import load_iris
from vulpes.automl import Clustering

dataset = load_iris()
X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])

clustering = Clustering()
df_models = clustering.fit(X)
df_models
```

### Fit a "best model"

We can automatically build a VotingClassifier or a VotingRegressor using the build_best_models method once the models are fitted.

```python
df_best = classifiers.build_best_models(X, y, nb_models=3)
df_best
```

|           Model | Balanced Accuracy | Accuracy | Precision |  Recall | F1 Score | Running time |
|----------------:|------------------:|---------:|----------:|--------:|---------:|-------------:|
| Voting (3-best) |           0.97508 | 0.974667 |  0.976034 | 0.97508 | 0.974447 |     11.82946 |

### Check missing data

```python
import pandas as pd
import numpy as np
df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", np.nan]],
                  dtype="category",
                  columns=["feature1", "feature2"])
classifiers.missing_data(df)
```

| Total Missing | Percentage (%) | Accuracy |
|--------------:|---------------:|---------:|
|    feature2   |              2 |     50.0 |
|    feature1   |              1 |     25.0 |

## Testing

If you want to submit a pull request or if you want to test in local the package, you can run some tests with the library pytest by running the following command:

```python
pytest vulpes/tests/
```

## Why Vulpes?

Vulpes stands for: **V**ector (**U**n)supervised **L**earning **P**rogram **E**stimation **S**ystem.

Nah, I'm kidding, I just love foxes, they are cute! The most common and widespread species of fox is the red fox (Vulpes vulpes).

![alt text](https://github.com/AdrienC21/vulpes/blob/main/fox.jpg?raw=true)

## Acknowledgment

- Shankar Rao Pandala (and some contributors). Their package (Lazy Predict) has been an inspiration.

## License

[MIT](https://choosealicense.com/licenses/mit/)
