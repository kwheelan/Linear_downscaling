
#==============================================================================
"""
A file to linearly downscale predictors.

Options:
    Lat-lon coordinate
    Predictand (obs) -- to do
    Predictors
    Start date -- to add
    End date -- to add
    Where to save the output
    LASSO or linear regression
    Monthly or annual model -- to do
    Condition or unconditional model -- to do
    Add stochasticity or not -- to do
    Scale of "variance inflation" (mean and variance of sampling distribution)
    Which plots to generate -- to do
    Which summary statistics to save to a file -- to do

Updated 7.28.2020, K. Wheelan

Usage: regress.py <lat> <lon> <obs filepath> <location to save data> <any pred file paths>+
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
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import os
import sys

with open("settings.txt") as f:
    settings = eval(f.read())

lat = sys.argv[1] #settings['lat']
lon = sys.argv[2] #settings['lon']
predictand = settings['predictand']
preds = settings['preds_surface'] + settings['preds_level']

#create a folder to save data
folderName = f"{predictand}_lat{lat}_lon{lon}"
ROOT = os.path.join(settings['save_path'],folderName)
try:
    os.mkdir(ROOT)
except FileExistsError:
    pass
settings['save_path'] = ROOT

print("Progress:")
print(f"Lat: {lat}, Lon: {lon}")

#import predictors
# default = ERA-I preds
if preds == ['all']:
    predictors = load_all_predictors()
else:
    predictors = load_selected_predictors(preds)
print("Loaded predictor files")

#==============================================================================
"""
#Clean up and prep the data for analysis.
"""
#==============================================================================


#standardize data, trim dates, add month and constant cols

X_all, Y_all = prep_data(settings['obs_path'], predictors, lat, lon,
                        dateStart = settings['dateStart'],
                        dateEnd = settings['dateEnd'])

if settings['transform']:
    #fourth root transformation (intended for precip)
    Y_all[predictand] = Y_all[predictand]**(1/4)

if settings['stdize']:
    #standardize predictors
    if settings['monthly']:
        #standardize data by month
        X_all = stdz_month(X_all)
    elif settings['apr_sep']:
        #standardize all data from apr-sep together
        X_all = stdz_subset(X_all) #todo: add range?
    else:
        X_all = standardize(X_all)


if settings['stdize_y']:
    Y_all, _ = stdz_month(Y_all)

#add necessary columns
X_all, Y_all, all_preds = add_month(X_all, Y_all)
X_all, all_preds = add_constant_col(X_all)

print("Loaded obs data.")


if settings['train']:
    #separate testing and training data by even and odd years
    X_train, X_test = evenOdd(X_all)
    Y_train, Y_test = evenOdd(Y_all)

    #creating a month filter
    for data in [X_train, X_test, Y_train, Y_test]:
        data = add_month_filter(data)
else:
    # train on all data
    X_all, Y_all = add_month_filter(X_all), add_month_filter(Y_all)

if settings['apr_sep']:
    #just for april - september data
    X_all['time'] = (X_all.time >= 4) & (X_all.time <= 9)
    Y_all['time'] = (Y_all.time >= 4) & (Y_all.time <= 9)
    X_all, Y_all = X_all.sel(time = True), Y_all.sel(time=True)
    X_all['time'], Y_all['time'] = X_all.timecopy, Y_all.timecopy

print("Prepped data for regression")


#==============================================================================
""" Calibrating regression model. """
#===============================================================================

# exclude variables that aren't predictors
preds_to_drop = ["month", "lat", "lon"]
preds_to_keep = [x for x in all_preds if not x in preds_to_drop]

if settings['train']:
    X = X_train
    y = Y_train
else:
    X = X_all
    y = Y_all

if settings['conditional']:
    logit_betas = fit_logistic(X, y, predictand)
    save_betas(settings, logit_betas, lat, lon, predictand, logistic = True)
    print("Fit conditional model")

#fit a different model for each month
if settings['method'] == 'OLS':
    if settings['monthly']:
        coefMatrix = fit_monthly_linear_models(X, y, preds_to_keep, predictand, settings['conditional'])
    else:
        coefMatrix = fit_annual_OLS(X, y, preds_to_keep, predictand, settings['conditional'])
elif settings['method'] == 'LASSO':
    if settings['monthly']:
        coefMatrix = fit_monthly_lasso_models(X, y, predictand, settings['conditional'])
    else:
        coefMatrix = fit_annual_lasso_model(X, y, predictand, settings['conditional'])

print("Fit linear model.")

#saves the betas
save_betas(settings, coefMatrix, lat, lon, predictand)
print("Saved betas.")
k = len([i for i in coefMatrix.iloc[:,0] if i != 0])

#==============================================================================
"""Generating predictions"""
#==============================================================================

#predict for all data using betas
if settings['conditional']:
    final_predictions = predict_conditional(X_all, coefMatrix, logit_betas, preds_to_keep, settings['stochastic_thresh'], settings['static_thresh'])
else:
    final_predictions = predict_linear(X_all, coefMatrix, preds_to_keep)
print("Calculated predictions for testing and training data.")

if settings['inflate']:
    # add stochasticity via "variance inflation", before undoing any data transformations
#    final_predictions = inflate_variance_SDSM(settings['inflate_mean'], settings['inflate_var'], final_predictions)
    final_predictions = inflate_variance_SDSM(y[predictand], final_predictions, c=settings['inflate_var'])

if settings['transform']:
    # undo transformation
    final_predictions['preds'] = final_predictions.preds ** 4


save_preds(settings['save_path'], final_predictions, lat, lon, predictand)
print("Saved predictions.")


#==============================================================================
"""Generate plots."""
#==============================================================================

if settings['transform']:
    #undoing fourth root transformation (intended for precip)
    Y_all[predictand] = Y_all[predictand]**4

fourth = final_predictions
fourth.preds = fourth.preds ** 4
quarter = final_predictions
quarter.preds = quarter.preds ** (0.25)

plotData = Plot(settings['save_path'], lat, lon, predictand, obs = Y_all,
                models = {'ERA-I': final_predictions,
                'ERA-I ^ 4': fourth ,
                'ERA-I ^ 0.25': quarter},
                startDate = settings['dateStart'],
                endDate = settings['dateEnd'], k = k)

for folder in ['seasonalPlots', 'distributionPlots', 'timeSeriesPlots']:
    try:
        os.mkdir(os.path.join(plotData.plot_path, folder))
    except: pass

plot_all(plotData)

print("Generated plots.")

print("Done.")
