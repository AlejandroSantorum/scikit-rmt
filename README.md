[![PyPI](https://img.shields.io/pypi/v/scikit-rmt?color=g)](https://pypi.org/project/scikit-rmt/)
[![Documentation Status](https://readthedocs.org/projects/scikit-rmt/badge/?version=latest)](https://scikit-rmt.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/AlejandroSantorum/scikit-rmt.svg?branch=main)](https://travis-ci.com/AlejandroSantorum/scikit-rmt)
[![codecov](https://codecov.io/gh/AlejandroSantorum/scikit-rmt/branch/main/graph/badge.svg?token=56TNEASPJK)](https://codecov.io/gh/AlejandroSantorum/scikit-rmt)
[![License Status](https://img.shields.io/github/license/AlejandroSantorum/scikit-rmt?color=blue)](https://github.com/AlejandroSantorum/scikit-rmt/blob/main/LICENSE)
[![PyPI-Python Version](https://img.shields.io/pypi/pyversions/scikit-rmt)](https://pypi.org/project/scikit-rmt)


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

First of all, several random matrix ensembles can be sampled: **Gaussian Ensembles**, **Wishart Ensembles**,
**Manova Ensembles** and **Circular Ensembles**. As an example, the following code shows how to sample
a **Gaussian Orthogonal Ensemble (GOE)** random matrix.

```python
from skrmt.ensemble import GaussianEnsemble
# sampling a GOE (beta=1) matrix of size 3x3
goe = GaussianEnsemble(beta=1, n=3)
print(goe.matrix)
```
```bash
[[ 0.34574696 -0.10802385  0.38245343]
 [-0.10802385 -0.60113963  0.28624612]
 [ 0.38245343  0.28624612 -0.96503739]]
```
Its spectral density can be easily plotted:
```python
# sampling a GOE matrix of size 1000x1000
goe = GaussianEnsemble(beta=1, n=1000)
# plotting its spectral distribution in the interval (-2,2)
goe.plot_eigval_hist(bins=80, interval=(-2,2), density=True)
```
![GOE density plot](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/hist_goe.png)
<!---
<img src="imgs/hist_goe.png" width=450 height=320 alt="GOE density plot">
-->

If we sample a **non-symmetric/non-hermitian** random matrix, its eigenvalues do not need to be real,
so a **2D complex histogram** has been implemented in order to study spectral density of these type
of random matrices. It would be the case, for example, of **Circular Symplectic Ensemble (CSE)**.

```python
# sampling a CSE (beta=4) matrix of size 2000x2000
cse = CircularEnsemble(beta=4, n=1000)
cse.plot_eigval_hist(bins=80, interval=(-2.2,2.2))
```
![CSE density plot](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/hist_cse_smooth.png)
<!---
<img src="imgs/hist_cse_smooth.png" width=650 height=320 alt="CSE density plot">
-->

We can **boost histogram representation** using the results described by A. Edelman and I. Dumitriu
in *Matrix Models for Beta Ensembles* and by J. Albrecht, C. Chan, and A. Edelman in
*Sturm Sequences and Random Eigenvalue Distributions* (check references). Sampling certain
random matrices (**Gaussian Ensemble** and **Wishart Ensemble** matrices) in its **tridiagonal form**
we can speed up histogramming procedure. The following graphical simulation using GOE matrices
tries to illustrate it.
![Speed up by tridigonal forms](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/gauss_tridiag_sim.png)
<!---
<img src="imgs/gauss_tridiag_sim.png" width=820 height=370 alt="Speed up by tridigonal forms">
-->

In addition, several spectral laws can be analyzed using this library, such as Wigner's Semicircle Law,
Marchenko-Pastur Law and Tracy-Widom Law. The analytical probability density function can also be plotted
by using the `limit_pdf` argument.

Plot of **Wigner's Semicircle Law**, sampling a GOE matrix 5000x5000:
```python
from skrmt.ensemble import wigner_semicircular_law

wigner_semicircular_law(ensemble='goe', n_size=5000, bins=80, density=True)
```
![Wigner Semicircle Law](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/scl_goe.png)
<!---
<img src="imgs/scl_goe.png" width=450 height=320 alt="Wigner Semicircle Law">
-->

```python
from skrmt.ensemble import wigner_semicircular_law

wigner_semicircular_law(ensemble='goe', n_size=5000, bins=80, density=True, limit_pdf=True)
```
![Wigner Semicircle Law PDF](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/scl_goe_pdf.png)
<!---
<img src="imgs/scl_goe_pdf.png" width=450 height=320 alt="Wigner Semicircle Law PDF">
-->

Plot of **Marchenko-Pastur Law**, sampling a WRE matrix 5000x5000:
```python
from skrmt.ensemble import marchenko_pastur_law

marchenko_pastur_law(ensemble='wre', p_size=5000, n_size=15000, bins=80, density=True)
```
![Marchenko-Pastur Law](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/mpl_wre.png)
<!---
<img src="imgs/mpl_wre.png" width=450 height=320 alt="Marchenko-Pastur Law">
-->

```python
from skrmt.ensemble import marchenko_pastur_law

marchenko_pastur_law(ensemble='wre', p_size=5000, n_size=15000, bins=80, density=True, limit_pdf=True)
```
![Marchenko-Pastur Law PDF](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/mpl_wre_pdf.png)
<!---
<img src="imgs/mpl_wre_pdf.png" width=450 height=320 alt="Marchenko-Pastur Law PDF">
-->

Plot of **Tracy-Widom Law**, sampling 20000 GOE matrices of size 100x100:
```python
from skrmt.ensemble import tracy_widom_law

tracy_widom_law(ensemble='goe', n_size=500, times=20000, bins=80, density=True)
```
![Tracy-Widom Law](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/twl_goe.png)
<!---
<img src="imgs/twl_goe.png" width=450 height=320 alt="Tracy-Widom Law">
-->

```python
from skrmt.ensemble import tracy_widom_law

tracy_widom_law(ensemble='goe', n_size=500, times=20000, bins=80, density=True, limit_pdf=True)
```
![Tracy-Widom Law PDF](https://raw.githubusercontent.com/AlejandroSantorum/scikit-rmt/main/imgs/twl_goe_pdf.png)
<!---
<img src="imgs/twl_goe_pdf.png" width=450 height=320 alt="Tracy-Widom Law PDF">
-->

The other module of this library implements **several covariance matrix estimators**:
* Sample estimator.
* Finite-sample optimal estimator (FSOpt estimator).
* Non-linear shrinkage analytical estimator (Ledoit & Wolf, 2020).
* Linear shrinkage estimator (Ledoit & Wolf, 2004).
* Empirical Bayesian estimator (Haff, 1980).
* Minimax estimator (Stain, 1982).

For certain problems, sample covariance matrix is not the best estimation for the
population covariance matrix.

The following code illustrates the usage of the estimators.
```python
from skrmt.covariance import analytical_shrinkage_estimator

# load dataset with your own/favorite function (such as pandas.read_csv)
X = load_dataset('dataset_file.data')

# get estimation
Sigma = analytical_shrinkage_estimator(X)

# ... Do something with Sigma. For example, PCA.
```

For more information or insight about the usage of the library, you can visit the official **documentation** 
<https://scikit-rmt.readthedocs.io/en/latest/> or the directory [notebooks](notebooks), that contains several
*Python notebooks* with **tutorials** and plenty of **examples**.

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
## Attribution
This project has been developed by Alejandro Santorum Varela (2021) as part of the final degree project
in Computer Science (Autonomous University of Madrid), supervised by Alberto Suárez González.

If you happen to use `scikit-rmt` in your work or research, please cite its GitHub repository:

A. Santorum, "scikit-rmt", https://github.com/AlejandroSantorum/scikit-rmt, 2021. GitHub repository.

The corresponding BibTex entry is
```
@misc{Santorum2021,
  author = {A. Santorum},
  title = {scikit-rmt},
  year = {2021},
  howpublished = {\url{https://github.com/AlejandroSantorum/scikit-rmt}},
  note = {GitHub repository}
}
```



