import os
import codecs

from setuptools import setup, find_packages


def read(filename):
    """Read and return `filename` in root dir of project and return string"""
    here = os.path.abspath(os.path.dirname(__file__))
    return codecs.open(os.path.join(here, filename), 'r').read()


install_requires = read("requirements.txt").split()
long_description = read('README.md')


setup(
    name="pyrallelsa",
    version="0.1.0",
    url='https://github.com/mesos-magellan/pyrallelsa',
    license='MIT License',
    author='Hamza Faran',
    description=('Framework for local parallel simulated annealing'),
    long_description=long_description,
    packages=find_packages(),
    install_requires = install_requires
)
