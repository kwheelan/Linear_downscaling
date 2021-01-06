
#==============================================================================
"""
A file to generate time series from existing betas.

Updated 1.6.2021, K. Wheelan

settings.txt should match the settings used to generate the betas
(to match explanatory vars, standardization, transformations, etc)

Usage: gen_time_series.py <lat> <lon> <location to save data> <beta location>
                          <pred root> <pred ext> <pred series>
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
import sys

with open("settings.txt") as f:
    settings  = eval(f.read())

lat = sys.argv[1] #settings['lat']
lon = sys.argv[2] #settings['lon']
save_location = sys.argv[3]
beta_location = sys.argv[4]
pred_root = sys.argv[5]
pred_ext = sys.argv[6]
pred_series = sys.argv[7]

predictand = settings['predictand']
preds = settings['preds_surface'] + settings['preds_level']

#import predictors
if preds == ['all']:
    predictors = load_all_predictors()
else:
    predictors = load_selected_predictors(preds,
                                          ROOT = pred_root,
                                          EXT = pred_ext,
                                          SERIES = pred_series)

#==============================================================================
"""
#Clean up and prep the data to match model.
"""
#==============================================================================


#standardize data, trim dates, add month and constant cols
X_all, Y_all = prep_data(settings['obs_path'], predictors, lat, lon, dateStart = settings['dateStart'], dateEnd = settings['dateEnd'])

#match standardization of original model
if settings['stdize']:
    #standardize predictors
    if settings['monthly']:
        #standardize data by month
        X_all = stdz_month(X_all)
    elif settings['apr_sep']:
        #standardize all data from apr-sep together
        X_all = stdz_subset(X_all)
    else:
        X_all = standardize(X_all)

#add necessary columns
X_all, _, all_preds = add_month(X_all, Y_all)
X_all, all_preds = add_constant_col(X_all)
X_all = add_month_filter(X_all)

#just for april - september data
#X_all['time'] = (X_all.time >= 4) & (X_all.time <= 9)
#X_all, Y_all = X_all.sel(time = True), Y_all.sel(time=True)
#X_all['time'] = X_all.timecopy

preds_to_drop = ["month", "lat", "lon"]
preds_to_keep = [x for x in all_preds if not x in preds_to_drop]

#==============================================================================
"""Generating predictions"""
#==============================================================================

#read in betas
coefMatrix = pd.read_csv(beta_location)
print(coefMatrix)

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
