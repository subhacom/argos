# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-04 4:13 PM
"""How to build distribution:

- Build wheel and tar (update MICRO-version number for test - once used, a filename cannot be reused)

      python -m build

- Check dist folder for old archives - delete them, rename files if required. Then upoload

      python -m twine upload --repository testpypi dist/*

      python -m twine upload --repository pypi dist/*

- Install from TestPyPI

      pip install --index-url https://test.pypi.org/simple/ --no-deps argos-toolkit --extra-index-url https://pypi.org/simple


``argos-tracker`` is just PyPI name, the installed package is named
``argos``. This is to avoid name conflict with another package on
PyPI.

There is an existing, unrelated ``argos`` package for viewing HDF5
files. If you must use both, see this:
https://stackoverflow.com/questions/27532112/how-to-handle-python-packages-with-conflicting-names

"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import re
import io
import os

# import argos - this import fails when running pip -e .


def read(*names):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names), encoding='utf8'
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version = read(*file_paths)
    version_match = re.search(
        r'^__version__ = ["\']([^"\']*)["\']', version, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string')


with open('README.md', 'r') as fh:
    long_description = fh.read()


ext_modules = [
    Extension(
        'yolact.utils.cython_nms',
        sources=['yolact/utils/cython_nms.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[],
    ),
    Extension(
        'argos.cutility',
        sources=['argos/cutility.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[],
    ),
    Extension(
        'argos.ccapture',
        sources=['argos/ccapture.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[],
    ),
]
setup(
    name='argos_toolkit',
    version=find_version('argos', '__init__.py'),  # argos.__version__,
    author='Subhasis Ray',
    author_email='ray.subhasis@gmail.com',
    description='Software tools to facilitate tracking multiple objects (animals) in a video.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/subhacom/argos',
    project_urls={
        'Documentation': 'https://argos.readthedocs.io',
        'Source': 'https://github.com/subhacom/argos',
        'Tracker': 'https://github.com/subhacom/argos/issues',
    },
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Public Domain',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
    ],
    python_requires='>=3.6',
    install_requires=[
        'cython',
        'torch',
        'torchvision',
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'tables',
        'sortedcontainers',
        'pyqt5',
        'opencv-contrib-python',
        'pyyaml',
        'matplotlib',
    ],
)
