import os
import re
import sys
import platform
import subprocess
from subprocess import CalledProcessError

import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


from os import path
this_directory = path.abspath(path.dirname(__file__))
import io
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


__version__ = '0.2.1'

kwargs = dict(
    name='resautonet',
    version=__version__,
    author='Lianfa Li',
    author_email='lspatial@gmail.com',
    description='Library for Autoencoder-based Residual Deep Network',
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    packages=find_packages(),
    install_requires = [],
    include_package_data=True,
    classifiers=[
        'Development Status :: 6 - Mature',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
#    package_data={'samples':['resautonet/data/*.csv']}
    data_files=[('csv', ['resautonet/data/pm25selsample.csv', 'resautonet/data/simdata.csv'])],
)

# likely there are more exceptions, take a look at yarl example 
try: 
    setup(**kwargs)   
except CalledProcessError: 
    print('Failed to build extension!') 
    del kwargs['ext_modules'] 
    setup(**kwargs) 

