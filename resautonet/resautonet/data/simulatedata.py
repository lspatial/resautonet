import numpy as np
import numpy.random as nr
import pandas as pd

def simData(n=1200):
    """
    Function to simulate the test dataset
    the simulated dataset in the format of Pandas's Data Frame,
          columns: ['x1','x2','x3','x4','x5','x6','x7','x8','y']
          (xi is the covariate and 'y' is the target variable)
          The simulated dataset generated according to the following formula:
             y=x1+x2*np.sqrt(x3)+x4+np.power((x5/500),0.3)-x6+np.sqrt(x7)+x8+noise
          each covariate is defined as:
                x1=nr.uniform(1,100,n)
                x2=nr.uniform(0,100,n)
                x3=nr.uniform(1,10,n)
                x4=nr.uniform(1,100,n)
                x5=nr.uniform(9,100,n)
                x6=nr.uniform(1,1009,n)
                x7=nr.uniform(5,300,n)
                x8=nr.uniform(6,200,n)
    :param n: integer, size of the sample
    :return: pandas's DataFrame object, with the colums,
             ['x1','x2','x3','x4','x5','x6','x7','x8','y']
    """
    x1=nr.uniform(1,100,n)
    x2=nr.uniform(0,100,n)
    x3=nr.uniform(1,10,n)
    x4=nr.uniform(1,100,n)
    x5=nr.uniform(9,100,n)
    x6=nr.uniform(1,1009,n)
    x7=nr.uniform(5,300,n)
    x8=nr.uniform(6,200,n)
    mu, sigma = 0, 90 # mean and standard deviation
    noise = np.random.normal(mu, sigma, n)
    y=x1+x2*np.sqrt(x3)+x4+np.power((x5/500),0.3)-x6+np.sqrt(x7)+x8+noise
    xydataDf=pd.DataFrame(np.stack([x1,x2,x3,x4,x5,x6,x7,x8,y],axis=1),
                          columns=['x1','x2','x3','x4','x5','x6','x7','x8','y'])
    return xydataDf