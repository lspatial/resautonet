# -*- coding: utf-8 -*-
""" Library of Autoencoder-based Residual Deep Network

This package provides the class of Autoencoder-based Residual Deep
Network with two sets of test datasets. Currently this library just
provides the support for Keras and may extend to other software of
deep learning later.

Major modules:
    model: including the major class of auto-encoder-based residual
           deep network, mainly for fully-linked neural network. Will
           extend to other networks like UNet in the future with the
           relevant metrics for regression of continuous variables.
           major class: resAutoencoder;
           rsquared and rmse metrics functions.
    peranalysis: parallel performance analysis for various arguments
           configures including network structure, action functions,
           output type, batch normalization, regularizer, residual
           connection, activation of the final output layer, mini
           batch size.
           major class: mulParPerAnalysis
    data: test data embedded in the package. We provide two datasets,
          one is the simulated dataset and the other is incomplete
          real dataset of PM2.5 with their covariates. This data is
          the 2015 data for the Beijing-Tianjing-Tangshan area of China.

Github source: https://github.com/lspatial/
Author: Lianfa Li
Date: 2018-10-01

"""

#import pkgutil
#__path__ = pkgutil.extend_path(__path__, __name__)

from resautonet.data.data import data
from resautonet.data.simulatedata import simData
from resautonet import model
from resautonet.peranalysis import perAnaCls
# from ._metrics import *
