.. scikit-rmt documentation master file, created by
   sphinx-quickstart on Wed Mar 31 11:38:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scikit-rmt documentation!
====================================

Random Matrix Theory, or RMT, is the field of Statistics that analyses
matrices that their entries are random variables.

This package offers classes, methods and functions to give support to RMT
in Python. Includes a wide range of utils to work with different random
matrix ensembles, random matrix spectral laws and estimation of covariance
matrices. See documentation or visit the `project page <https://github.com/AlejandroSantorum/scikit-rmt>`_
hosted by Github for further information on the features included in the package.

.. toctree::
   :caption: Using scikit-rmt
   :hidden:
	
   auto_tutorial/index

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   docs/skrmt
   docs/skrmt.ensemble
   docs/skrmt.covariance

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`


Installation
============

Using a virtual environment is recommended to minimize the chance of conflicts.
However, the global installation should work properly as well.

Local installation using `venv` (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to your project directory.

.. code:: bash

	cd MyProject

Create a virtual environment (you can change the name "env").

.. code:: bash

	python3 -m venv env

Activate the environment "env".

.. code:: bash

	source env/bin/activate

Install using `pip`.

.. code:: bash

	pip install scikit-rmt

You may need to use `pip3`.

.. code:: bash

	pip3 install scikit-rmt

Global installation
~~~~~~~~~~~~~~~~~~~

Just install it using `pip`or `pip3`.

.. code:: bash

	pip install scikit-rmt

Requirements
============

*scikit-rmt* depends on the following packages:

* `numpy <https://github.com/numpy/numpy>`_ - The fundamental package for scientific computing with Python
* `matplotlib <https://github.com/matplotlib/matplotlib>`_ - Plotting with Python
* `scipy <https://github.com/scipy/scipy>`_ - Scientific computation in Python

