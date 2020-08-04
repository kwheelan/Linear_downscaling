
#==============================================================================
"""A script to process and plot SDSM output for comparison."""
#==============================================================================

#import packages

import xarray as xr
import pandas as pd
import numpy as np
from plotting import *
import warnings
warnings.filterwarnings('ignore')

lat = 38.125
lon = -101.875
obsPath = '/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc'
save_path = '/glade/scratch/kwheelan/downscaling_data/SDSM_p1'
pred_file = '../SDSM_p1/tmax_p1_output.OUT'
predictand = 'tmax'
dateStart = '1980-01-01'
dateEnd = '2005-12-31'

# read in predictions from an "OUT" file
f = open(pred_file, 'r')
sdsm_preds = [float(line.strip()) for line in f]
f.close()

# get obs netCDF
obs = xr.open_dataset(obsPath).sel(time = slice(dateStart, dateEnd))
Y_all = obs.sel(lat = lat, lon = lon, method = 'nearest').sel(time = slice(dateStart, dateEnd))
preds = xr.open_dataset('/glade/scratch/kwheelan/downscaling_data/preds/finalPreds_tmax_38.125_-101.875.nc')
SDSM = preds.copy()

#set up variables
Y_all['timecopy'] = Y_all['time']
Y_all['month'] = Y_all.time.dt.month
SDSM['preds'] = ({'time':'time'}, sdsm_preds)

plotData = Plot(".", lat, lon, predictand, obs = Y_all, models = {'Python': preds, 'SDSM': SDSM})

plot_all_seasons(plotData)
plot_monthly_avgs(plotData)
if predictand == 'tmax':
    plot_hot_days(plotData)
elif predictand == 'tmin':
    plot_cold_days(plotData)
save_stats(plotData)
plot_dists(plotData)
