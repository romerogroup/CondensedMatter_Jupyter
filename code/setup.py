#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='minimulti',
    version='0.3',
    description='Mini Extendable framework of multi Hamiltonian',
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    package_data={},
    install_requires=['numpy', 'scipy',  'matplotlib', 'ase', 'numba', 'phonopy', 'netcdf4','ipyvolumn'
        ],
    scripts=[
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
    ])
