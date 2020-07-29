
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

TODO:
add a switch for conditional regression (precip)
add a switch for monthly models
add a switch for testing/train
allow predictand to be something other than tmax
allow options saved in separate file
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


#set variables based on commandline arguments
if len(sys.argv) < 5:
    exit("Usage: regress.py <lat> <lon> <obsPath> <save_path> <preds>+")

lat = float(sys.argv[1])
lon = float(sys.argv[2])
obsPath = sys.argv[3] #filepath for obs data
save_path = sys.argv[4]
preds = [ path for path in sys.argv[5:] ] #paths for preds files

#to be customized:
dateStart = '1980-01-01'
dateEnd = '2005-12-31'
method = "OLS"
train = False
stdize = True
inflate = False

print("Progress:")
print(f"Lat: {lat}, Lon: {lon}")

#import predictors
if preds == ['all']:
    predictors = load_all_predictors()
else:
    predictors = load_selected_predictors(preds)
print("Loaded predictor files")

#==============================================================================
"""
Clean up and prep the data for analysis.
"""
#==============================================================================


#standardize data, trim dates, add month and constant cols

print(f"mslp: mean: {np.mean(predictors.mslp)} var: {np.var(predictors.mslp)}\n")

if stdize:
    predictors = standardize(predictors)
X_all, Y_all = prep_data(obsPath, predictors, lat, lon, dateStart = dateStart, dateEnd = dateEnd)
X_all, Y_all, all_preds = add_month(X_all, Y_all)
X_all, all_preds = add_constant_col(X_all)
print("Loaded obs data.")

print(f"predictors mslp: mean: {np.mean(predictors.mslp)} var: {np.var(predictors.mslp)}\n")
print(f"X_all stdized mslp: mean: {np.mean(X_all.mslp)} var: {np.var(X_all.mslp)}\n")

if train:
    #separate testing and training data by even and odd years
    X_train, X_test = evenOdd(X_all)
    Y_train, Y_test = evenOdd(Y_all)

    #creating a month filter
    for data in [X_train, X_test, Y_train, Y_test]:
        data = add_month_filter(data)
else:
    # train on all data
    X_all, Y_all = add_month_filter(X_all), add_month_filter(Y_all)

print("Prepped data for regression")


#==============================================================================
""" Calibrating regression model. """
#===============================================================================

# exclude variables that aren't predictors
preds_to_drop = ["month", "lat", "lon"]
preds_to_keep = [x for x in all_preds if not x in preds_to_drop]

#fit a different model for each month
if train:
    if method == 'OLS':
        coefMatrix = fit_monthly_linear_models(X_train, Y_train, preds_to_keep)
    elif method == 'LASSO':
        coefMatrix = fit_monthly_lasso_models(X_train, Y_train)
else:
    if method == 'OLS':
        coefMatrix = fit_monthly_linear_models(X_all, Y_all, preds_to_keep)
    elif method == 'LASSO':
        coefMatrix = fit_monthly_lasso_models(X_all, Y_all)
print("Fit linear model.")

#saves the betas
save_betas(save_path, coefMatrix, lat, lon)
print("Saved betas.")

#==============================================================================
"""Generating predictions"""
#==============================================================================

#predict for all data using betas
final_predictions = predict_linear(X_all, coefMatrix, preds_to_keep)
print("Calculated predictions for testing and training data.")

# TODO: transformations
if inflate:
    corrected_preds = inflate_variance(-0.5, 4, final_predictions)

save_preds(save_path, final_predictions, lat, lon)
print("Saved predictions.")


#==============================================================================
"""Generate plots."""
#==============================================================================

plot_all_seasons(Y_all, final_predictions, save_path, lat, lon)
plot_monthly_avgs(Y_all, final_predictions, save_path, lat, lon)
plot_hot_days(Y_all, final_predictions, save_path, lat, lon)
save_stats(Y_all, final_predictions, lat, lon, save_path)

print("Generated plots.")

print("Done.")
