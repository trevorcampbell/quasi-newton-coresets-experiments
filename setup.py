from setuptools import setup, find_packages

setup(
    name = 'fast-bayesian-coresets',
    description="Fast Bayesian Coresets via Subsampling and Quasi-Newton Refinement",
    packages=find_packages(exclude=('examples','examples.*')),
    install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'bokeh', 'pystan'],
    platforms='ALL',
)
