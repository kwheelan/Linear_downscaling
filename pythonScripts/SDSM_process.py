
#==============================================================================
"""A script to process and plot SDSM output for comparison."""
#==============================================================================

#import packages
import xarray as xr
import pandas as pd
import numpy as np
from plotting import *

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

#set up variables
Y_all['timecopy'] = Y_all['time']
Y_all['month'] = Y_all.time.dt.month
preds['preds'] = ({'time':'time'}, sdsm_preds)

# generate plots
plot_all_seasons(Y_all, preds, save_path, lat, lon, predictand)
plot_monthly_avgs(Y_all, preds, save_path, lat, lon, predictand)
plot_hot_days(Y_all, preds, save_path, lat, lon)
save_stats(Y_all, preds, lat, lon, save_path, predictand)
plot_dist(Y_all[predictand], 'Observed Distribution', save_path, lat, lon, predictand)
plot_dist(preds.preds, 'SDSM Modeled Distribution', save_path, lat, lon, predictand)
