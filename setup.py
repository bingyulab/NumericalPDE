
from setuptools import setup, find_packages

setup(
    name='poisson_solver',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'networkx'
    ],
    entry_points={
        'console_scripts': [
            'poisson_solver=main:validate_implementation',
        ],
    },
)