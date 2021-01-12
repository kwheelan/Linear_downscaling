
#==============================================================================
"""
A file to generate time series from existing betas.

Updated 1.8.2021, K. Wheelan

settings.txt should match the settings used to generate the betas
(to match explanatory vars, standardization, transformations, etc)

Usage: gen_time_series.py <lat> <lon> <location to save data> <beta location>
                          <pred root> <pred ext> <pred series>
"""
#==============================================================================

#todo: split settings from metadata
# save std values somewhere


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
import sys

lat = sys.argv[1]
lon = sys.argv[2]
save_location = sys.argv[3]
beta_location = sys.argv[4]
pred_root = sys.argv[5]
pred_ext = sys.argv[6]
pred_series = sys.argv[7]
start_date = sys.argv[8]
end_date = sys.argv[9]

with open(f"{beta_location}/metadata.txt") as f:
    settings  = eval(f.read())

predictand = settings['predictand']
preds = settings['preds_surface'] + settings['preds_level']

# if True, this is a future climate projection
future = int(start_date[0:4]) > 2020

folderName = f"{predictand}_lat{lat}_lon{lon}"
ROOT = os.path.join(save_location, folderName)
try:
    os.mkdir(ROOT)
except FileExistsError:
    pass
save_location = ROOT


#import predictors
if preds == ['all']:
    predictors = load_all_predictors()
else:
    if future:
        surf_ext = ''
    else: surf_ext = '_surf'
    predictors = load_selected_predictors(preds,
                                          ROOT = pred_root,
                                          EXT = pred_ext,
                                          SERIES = pred_series,
                                          surf_ext = surf_ext)

#==============================================================================
"""
#Clean up and prep the data to match model.
"""
#==============================================================================


#standardize data, trim dates, add month and constant cols
# obs start in 1980
X_all, Y_all = prep_data(settings['obs_path'], predictors, lat, lon,
                        start_date, end_date)

#match standardization of original model
if settings['stdize']:
    #standardize predictors
    if settings['monthly']:
        #standardize data by month
        path = f"{ROOT}/anom.txt".replace('rcp85', 'historical')
        if not future:
            X_all = stdz_month(X_all, anomSavePath = path)
        else:
            X_all = stdz_month(X_all, base_values = path)

    elif settings['apr_sep']:
        #standardize all data from apr-sep together
        X_all = stdz_subset(X_all)
    else:
        X_all = standardize(X_all)

#add necessary columns
X_all, Y_all, all_preds = add_month(X_all, Y_all)
X_all, all_preds = add_constant_col(X_all)
X_all = add_month_filter(X_all)

preds_to_drop = ["month", "lat", "lon"]
preds_to_keep = [x for x in all_preds if not x in preds_to_drop]

#==============================================================================
"""Generating predictions"""
#==============================================================================

#read in betas
coefMatrix = pd.read_csv(f"{beta_location}/betas.txt", index_col=0)

#predict for all data using betas
if settings['conditional']:
    final_predictions = predict_conditional(X_all, coefMatrix, logit_betas, predictand, glm, preds_to_keep, thresh = settings['static_thresh'], )
else:
    final_predictions = predict_linear(X_all, coefMatrix, preds_to_keep)

if settings['inflate']:
    # add stochasticity via "variance inflation", before undoing any data transformations
    final_predictions = inflate_variance_SDSM(y[predictand], final_predictions, c=settings['inflate_var'])

if settings['transform']:
    # undo transformation
    final_predictions['preds'] = final_predictions.preds ** 4


save_preds(save_location, final_predictions, lat, lon, predictand)

#=================================================================
""" Make plots for future and historical GCM runs """
#=================================================================

k = len([i for i in coefMatrix.iloc[:,0] if i != 0])
Y_all['timecopy'] = Y_all['time']

for folder in ['seasonalPlots', 'distributionPlots', 'timeSeriesPlots']:
    try:
        os.mkdir(os.path.join(plotData.plot_path, folder))
    except: pass


if not future:
    #historical
    #plot against obs 1980-2005
    plotData = Plot(save_location, lat, lon, predictand, obs = Y_all,
                models = {'downscaled GCM': final_predictions.sel(time=slice('1980-01-01', end_date))},
                startDate = '1980-01-01',
                endDate = end_date, k = k)
    plot_all(plotData)

else:
    pass
    #future projections
    #set years equal to plot
