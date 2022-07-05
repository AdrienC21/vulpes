import vulpes

from setuptools import setup, find_packages

VERSION = vulpes.__version__
DESCRIPTION = ("Test many classification, regression models and "
               "clustering algorithms to see which one is most "
               "suitable for your dataset.")
# LONG_DESCRIPTION: README + CHANGELOG
LONG_DESCRIPTION = ""
with open("README.md", "r+", encoding="UTF-8") as f:
    LONG_DESCRIPTION += f.read()
LONG_DESCRIPTION += "\n\n"
with open("CHANGELOG.rst", "r+") as f:
    LONG_DESCRIPTION += f.read()
# Install Requires
with open("requirements.txt", "r+") as f:
    INSTALL_REQUIRES = [x.replace("\n", "") for x in f.readlines()]

URL = "https://vulpes.readthedocs.io/en/latest/"
DOWNLOAD_URL = "https://pypi.org/project/vulpes/"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/AdrienC21/vulpes/issues",
    "Documentation": URL,
    "Source Code": "https://github.com/AdrienC21/vulpes",
}
AUTHOR = "Adrien Carrel"
AUTHOR_EMAIL = "a.carrel@hotmail.fr"
PYTHON_REQUIRES = ">=3.7"
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Unix',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules'
]
KEYWORDS = ["vulpes", "python", "automl", "scikit-learn",
            "machine-learning", "machine", "learning", "model",
            "artificial", "intelligence", "clustering",
            "dataset", "classification", "regression",
            "hyperparameter-tuning"]

setup(
    name="vulpes",
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    license="MIT",
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(),
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES
)
