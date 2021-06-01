import pkg_resources
import os.path
import pandas as pd

def data(name):
    """
    Function to obtain the sample data.
    :param name: string, the name for each of two datasets.
          'sim': simulated dataset in the format of Pandas's Data Frame,
                  columns: ['x1','x2','x3','x4','x5','x6','x7','x8','y']
                  (xi is the covariate and 'y' is the target variable)
                  You can also call the function in the data module,
                    simData to obtain the new simulated data.
                  The simulated dataset generated according to the following formula:
                     y=x1+x2*np.sqrt(x3)+x4+np.power((x5/500),0.3)-x6+np.sqrt(x7)+x8+noise
                  each covariate defined as:
                        x1=nr.uniform(1,100,n)
                        x2=nr.uniform(0,100,n)
                        x3=nr.uniform(1,10,n)
                        x4=nr.uniform(1,100,n)
                        x5=nr.uniform(9,100,n)
                        x6=nr.uniform(1,1009,n)
                        x7=nr.uniform(5,300,n)
                        x8=nr.uniform(6,200,n)
          'pm2.5':string, the name for a real dataset of the 2015 PM2.5 and the relevant covariates for
                    the Beijing-Tianjin-Tangshan area. It is sampled by the fraction of 0.8 from
                    the original dataset (stratified by the julian day). The covariates are defined
                    as the following:
                    'lat': latitude;
                    'lon': longitude;
                    'ele': elevation;
                    'prs': precipitation (mm);
                    'tem':air temperature (oC);
                    'rhu': relative humidity ;
                    'win': wind speed (m/s);
                    'aod': target variable,
                           Multi-Angle Implementation of Atmospheric Correction
                           Aerosol Optical Depth  (MAIAC AOD);
                    'pblh_re': Planetary boundary layer height at course resolution from NASA
                    'pre_re': Precipitation extrated from the images at course resolution from NASA
                    'o3_re': Ozone extraced from the images at course resolution from NASA
                    'merra2_re':MERRA2 AOD from the images at course resolution from NASA
                    'haod': Yearly averages for MAIAC AOD
                    'shaod': Monthly averages for MAIAC AOD
                    'jd': index for julian day
    :return: pandas's DataFrame, the dataset of the sample data for test
    """
    fname=name
    if name=='pm2.5':
        fname='pm25selsample.csv'
    elif name=='sim':
        fname = 'simdata.csv'
    fl=pkg_resources.resource_filename(__name__, '/'+fname)
    if not os.path.isfile(fl):
        print('Data not exists, please enter the correct file name ("sim" or "pm2.5")')
        return None
    gdata=pd.read_csv(fl,index_col='index')
    return gdata
