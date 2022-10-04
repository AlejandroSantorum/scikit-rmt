# encoding: utf-8

import os
import sys

from setuptools import find_packages, setup

with open("VERSION", "r") as version_file:
    VERSION = version_file.read().strip()

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().split('\n')

DOWNLOAD_URL = f'https://github.com/AlejandroSantorum/scikit-rmt/archive/refs/tags/v{VERSION}.tar.gz'

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
    download_url=DOWNLOAD_URL,
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