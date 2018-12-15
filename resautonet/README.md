# Library of Autoencoder-based Residual Deep Network (resautonet)

[![Build Status](https://travis-ci.org/pybind/cmake_example.svg?branch=master)](https://travis-ci.org/pybind/cmake_example)
[![Build status](https://ci.appveyor.com/api/projects/status/57nnxfm4subeug43/branch/master?svg=true)](https://ci.appveyor.com/project/dean0x7d/cmake-example/branch/master)

The python library of autoencoder based residual deep network (resautonet). 
Current version just supports the KERAS package of deep learning and 
will extend to the others in the future. 

## Major modules

**model**

* resAutoencoder: major class to obtain a autoencoder-based residual 
      deep network by setting the arguments. See the class and its 
      member functions' help for details.  
* pmetrics: functions for regression metrics like rsquared and RMSE. 

**peranalysis**

* mulParPerAnalysis: major class for parallel performance analysis 
      You can setup many configure parameters for each network (a duty)
      and then run them to the effects in a parallel way. See this class 
      and its member functions' help for details.  

**data**

* data: function to access each of two datasets,  
         sim': simulated dataset in the format of Pandas's Data Frame,
         'pm2.5':string, the name for a real dataset of the 2015 PM2.5 
            and the relevant covariates for the Beijing-Tianjin-Tangshan
            area. It is sampled by the fraction of 0.8 from the
           the original dataset (stratified by the julian day).
         See this function's help for details.  
* simdata: function to simulate the test dataset,  
         The simulated dataset generated according to the formula:
             y=x1+x2*np.sqrt(x3)+x4+np.power((x5/500),0.3)-x6+
                np.sqrt(x7)+x8+noise
         See this function's help for details.

## Installation

You can directly install it using the following command for the latest version:
     pip install resautonet -U  
You can also clone the repository and then install:

```bash
git clone --recursive https://github.com/lspatial/resautonet.git
cd package 
pip install ./setup.py install 
```

With the `setup.py` file included in this example, the `pip install` command will
invoke CMake and build the resautonet module as specified in `CMakeLists.txt`.


## Note for installation and use 

**Compiler requirements**

resautonet requires a C++11 compliant compiler to be available.

**Runtime requirements**

resautonet requires installation of Keras with support of Tensorflow or other 
backend system of deep learning (to support Keras). Also Pandas and Numpy should 
be installed. 


## Use case 
The homepage of the github for the package, resautonet provides two specific 
examples for use of autoencoder based residual deep network:  
https://github.com/lspatial/resautonet 


## License

The resautonet is provided under a MIT license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Test call

```python
import resautonet as r
#Load the sample dataset for PM2.5  
simdata=r.data('pm2.5')
simdata.head()
```
## Collaboration

Welcome to contact Dr. Lianfa Li (Email: lspatial@gmail.com). 