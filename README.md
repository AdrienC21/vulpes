# Vulpes

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pypi version](https://img.shields.io/pypi/v/vulpes.svg)](https://pypi.python.org/pypi/vulpes)
[![Documentation Status](https://readthedocs.org/projects/vulpes/badge/?version=latest)](https://vulpes.readthedocs.io/en/latest/?badge=latest)
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

Test:

```python
test
```

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
