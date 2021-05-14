[![Documentation Status](https://readthedocs.org/projects/scikit-rmt/badge/?version=latest)](https://scikit-rmt.readthedocs.io/en/latest/?badge=latest)
[![License Status](https://img.shields.io/github/license/AlejandroSantorum/scikit-rmt)](https://img.shields.io/github/license/AlejandroSantorum/scikit-rmt)
[![codecov](https://codecov.io/gh/AlejandroSantorum/scikit-rmt/branch/main/graph/badge.svg?token=56TNEASPJK)](https://codecov.io/gh/AlejandroSantorum/scikit-rmt)

# scikit-rmt: Random Matrix Theory Python package

Random Matrix Theory, or RMT, is the field of Statistics that analyses
matrices that their entries are random variables.

This package offers classes, methods and functions to give support to RMT
in Python. Includes a wide range of utils to work with different random
matrix ensembles, random matrix spectral laws and estimation of covariance
matrices. See documentation or visit the <https://github.com/AlejandroSantorum/scikit-rmt>
of the project for further information on the features included in the package.

-----------------
## Documentation

The documentation is available at  <https://scikit-rmt.readthedocs.io/en/latest/>,
which includes detailed information of the different modules, classes and methods of
the package, along with several examples showing different funcionalities.

-----------------
## Installation

Using a virtual environment is recommended to minimize the chance of conflicts.
However, the global installation _should_ work properly as well.

### Local installation using `venv` (recommended)

Navigate to your project directory.
```bash
cd MyProject
```

Create a virtual environment (you can change the name "env").
```bash
python3 -m venv env
```

Activate the environment "env".
```bash
source env/bin/activate
```

Install using `pip`.
```bash
pip install scikit-rmt
```
You may need to use `pip3`.
```bash
pip3 install scikit-rmt
```

### Global installation
Just install it using `pip`or `pip3`.
```bash
pip install scikit-rmt
```

### Requirements
*scikit-rmt* depends on the following packages:
* [numpy](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python
* [matplotlib](https://github.com/matplotlib/matplotlib) - Plotting with Python
* [scipy](https://github.com/scipy/scipy) - Scientific computation in Python


-----------------
## A brief tutorial

TODO

```python
from skrmt.ensemble import GaussianEnsemble

goe = GaussianEnsemble(beta=1, n=100)
goe.plot_eigval_hist(bins=80, interval=(-2,2), density=True)
```

TODO

-----------------
## License
The package is licensed under the BSD 3-Clause License. A copy of the [license](LICENSE) can be found along with the code.

-----------------
## Main references
- James Albrecht, Cy Chan, and Alan Edelman,
    "Sturm Sequences and Random Eigenvalue Distributions",
    *Foundations of Computational Mathematics*,
    vol. 9 iss. 4 (2009), pp 461-483.
    [[pdf]](http://www-math.mit.edu/~edelman/homepage/papers/sturm.pdf)
    [[doi]](http://dx.doi.org/10.1007/s10208-008-9037-x)

- Ioana Dumitriu and Alan Edelman,
    "Matrix Models for Beta Ensembles",
    *Journal of Mathematical Physics*,
    vol. 43 no. 11 (2002), pp. 5830-5547
    [arXiv:math-ph/0206043](http://arxiv.org/abs/math-ph/0206043)

- Rowan Killip and Rostyslav Kozhan,
    "Matrix Models and Eigenvalue Statistics for Truncations of Classical Ensembles of Random Unitary Matrices",
    *Communications in Mathematical Physics*, vol. 349 (2017) pp. 991-1027.
    [arxiv.org/pdf/1501.05160.pdf](http://arxiv.org/pdf/1501.05160.pdf)

- Olivier Ledoit and Michael Wolf,
    "Analytical Nonlinear Shrinkage of Large-dimensional Covariance Matrices",
    *Annals of Statistics*, vol. 48, no. 5 (2020) pp. 3043–3065.
    [[pdf]](http://www.econ.uzh.ch/static/wp/econwp264.pdf)

- Olivier Ledoit and Michael Wolf,
    "A Well-conditioned Estimator for Large-dimensional Covariance Matrices",
    *Journal of Multivariate Analysis*, vol. 88 (2004) pp. 365–411.
    [[pdf]](http://www.ledoit.net/ole1a.pdf)

-----------------
## Author
This project has been developed by Alejandro Santorum Varela (2021) as part of the final degree project
in Computer Science (Autonomous University of Madrid), supervised by Alberto Suárez González.





