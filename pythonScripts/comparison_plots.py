#==============================================================================
"""
A file to graphically compare time series from existing timeseries.

Updated 1.26.2021, K. Wheelan

"""
#==============================================================================


#import functions from other files
from regression_methods import *
from plotting import *

#import necessary packages
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import sklearn
import pandas as pd
import numpy as np
import os

future = xr.open_dataset('/glade/work/kwheelan/Linear_downscaling/GCM_downscaled/rcp85/tmax_lat32.125_lon-101.875/timeseries/finalPreds_tmax_32.125_-101.875.nc')
historical = xr.open_dataset('/glade/work/kwheelan/Linear_downscaling/GCM_downscaled/historical/tmax_lat32.125_lon-101.875/timeseries/finalPreds_tmax_32.125_-101.875.nc')
#obs
Y_all = xr.open_dataset('/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc')

#match years for plotting
future['time'] =  int(Y_all.time[0]) - int(future.time[0]) + future.time
plotData = Plot(save_location, lat, lon, predictand, obs = Y_all,
            models = {'MPI future': future.sel(time=slice('1980-01-01', '2005-12-31')),
                      'MPI historical': historical.sel(time=slice('1980-01-01', '2005-12-31'))},
            startDate = '1980-01-01',
            endDate = '2005-12-31', k = 5)
plot_all_future(plotData)
