# encoding: utf-8

import os
import re
from setuptools import find_packages, setup

# Getting release version
VERSION_FILEPATH = "skrmt/_version.py"
with open(
    os.path.join(os.path.dirname(__file__), VERSION_FILEPATH),
    "r",
) as version_file:
    version_file_text = version_file.read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    match = re.search(VSRE, version_file_text, re.M)
    if match:
        VERSION = match.group(1)
    else:
        raise RuntimeError(f"Unable to find version in {VERSION_FILEPATH}.")

# Getting README to include as a long description
with open(
    os.path.join(os.path.dirname(__file__), "README.md"),
    "r",
) as readme_file:
    LONG_DESCRIPTION = readme_file.read()

# Getting packaging requirements from requirements.txt file
with open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"),
    "r",
) as req_file:
    REQUIREMENTS = req_file.read().split('\n')

# URL to download release
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
