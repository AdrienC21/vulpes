from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "Vulpes: Test many classification and regression models with or without hyperparameter tuning to see which one is most suitable for your dataset"
LONG_DESCRIPTION = open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read()
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
KEYWORDS = ["vulpes", "python", "machine", "learning", "model",
            "scikit", "learn", "artificial", "intelligence",
            "dataset", "classification", "regression",
            "hyperparameter", "tuning"]

setup(
  name='vulpes',
  version=VERSION,
  description=DESCRIPTION,
  long_description_content_type="text/markdown",
  long_description=LONG_DESCRIPTION,
  url='',  
  author='Adrien Carrel',
  author_email='a.carrel@hotmail.fr',
  maintainer_email='a.carrel@hotmail.fr',
  license='MIT', 
  classifiers=CLASSIFIERS,
  keywords=KEYWORDS, 
  packages=find_packages(),
  install_requires=[''] 
)
