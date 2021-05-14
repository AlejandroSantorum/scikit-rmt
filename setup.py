# encoding: utf-8

import os
import sys

from setuptools import find_packages, setup

VERSION = '0.1.3'

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().split('\n')

setup(
    name='scikit-rmt',
    author='Alejandro Santorum Varela',
    author_email='alejandro.santorum@gmail.com',
    #packages = ['scikit-rmt'],
    version = VERSION,
    license='BSD',
    description='Random Matrix Theory Python package',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url = 'https://github.com/AlejandroSantorum/scikit-rmt',
    download_url='https://github.com/AlejandroSantorum/scikit-rmt/archive/refs/tags/v0.1.3.tar.gz',
    include_package_data=True,
    packages=find_packages(),
    keywords=['RMT', 'Random Matrix Theory', 'Ensemble', 'Covariance matrices'],
    #install_requires=["numpy", "matplotlib", "scipy"],
    install_requires=REQUIREMENTS,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)