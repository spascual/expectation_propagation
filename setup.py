from setuptools import setup


install_requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'pandas>=0.18.1',
    'matplotlib>=2.2.2',
]

setup(
    name='Expectation-propagation',
    author='Sergio Pascual',
    install_requires=install_requirements
)