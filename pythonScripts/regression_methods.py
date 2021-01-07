
#==============================================================================
""""
A module of functions for regression/downscaling

K. Wheelan
July, 2020
"""
#================================================================================


__all__ = ['load_all_predictors', 'load_selected_predictors', 'zscore', 'standardize',
            'stdz_subset', 'stdz_month',
            'prep_data','add_month', 'add_constant_col', 'evenOdd', 'add_month_filter',
            'fit_linear_model', 'fit_monthly_linear_models', 'fit_monthly_lasso_models',
            'fit_annual_lasso_model', 'fit_annual_OLS',
            'fit_logistic', 'save_betas', 'predict_linear', 'predict_conditional',
            'save_preds', 'inflate_variance', 'inflate_variance_SDSM']


import xarray as xr
import sklearn
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import os
from random import random


#Set globals
monthsAbrev = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthsFull = ['January','February', 'March','April','May','June','July','August','September','October','November','December']
month_range = range(1,13) #range(4,10)
#todo: fix this for apr-sept

#==============================================================================
"""
Functions for loading files.
"""
#===============================================================================


def load_selected_predictors(preds, predictors = 'ERA-I', ROOT=None, EXT=None, SERIES=None):
    """
        Reading in selected predictors (netCDF files) as xarray objects
        Input:
            preds, an array of filepaths to the predictor files
            ROOT, EXT, SERIES to specify file names, which take the form:
                {ROOT}{variable}{SERIES}[{pressure level}]{EXT}
        Output:
            merged predictors as an xarray object
    """
    if predictors =='ERA-I' or not ROOT: #default to ERA-I predictors
        ROOT = '/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/' #where the files are saved
        EXT = '_19790101-20181231_dayavg_mpigrid.nc' #date range at the end of the file
        SERIES = 'ERAI_NAmerica'

    if predictors == 'GCM' :
        ROOT = '/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/MPI-ESM-LR/mpigrid/'
        EXT = '19500101-20051231_dayavg_mpigrid.nc'
        SERIES = 'MPI-ESM-LR_historical_r1i1p1_NAmerica'

    #The variables to use
    surface_predictors = ['mslp', 'uas', 'vas'] #, 'ps']
    surface_predictors = [f"{v}_{SERIES}_surf" for v in preds if v in surface_predictors]
    #Each of these predictors is taken at each pressure level below
    other_predictors = ['Q', 'RH', 'U', 'V', 'Z', 'Vort', 'Div']
    levels = [500, 700, 850] #pressure levels
    #all combinations of predictors and levels
    full = [f"{v}_p{level}" for v in other_predictors for level in levels ]
    # filter specfied predictors
    level_preds = [f"{v.split('_')[0]}_{SERIES}_{v.split('_')[1]}" for v in preds if v in full]
    preds_long = surface_predictors + level_preds

    #Surface predictors
    for i in range(len(preds)):
        file = ROOT + preds_long[i] + EXT
        var = preds[i]
        if i == 0:
            predictors = xr.open_dataset(file)[var.split('_')[0]]
        # add new col and rename using level
        predictors = xr.merge([predictors, xr.open_dataset(file)[var.split('_')[0]]]).rename({var.split('_')[0]: preds[i]})

    return predictors

def load_all_predictors():
    """
        Read in predictors from specified files.
        Input: None (change to filenames)
        Output: Merged predictors as an xarray object
    """
    surface_predictors = ['mslp', 'uas', 'vas'] #, 'ps']
    #Each of these predictors is taken at each pressure level below
    other_predictors = ['Q', 'RH', 'U', 'V', 'Z', 'Vort', 'Div']
    levels = [500, 700, 850] #pressure levels
    level_preds = [f"{v}_p{level}" for v in other_predictors for level in levels]

    return load_selected_predictors(surface_predictors + level_preds)




#==============================================================================
"""
Functions for prepping data for regression.
"""
#==============================================================================

def zscore(variable, mu = None, sd = None):
    """
        Standardizing a variable (converting to z-scores)
        Input: an Xarray dataset
        Output: a standardized Xarray object
    """
    if not mu:
        mu = np.mean(variable)
    if not sd:
        sd = np.std(variable)
    return (variable - mu) / sd


def standardize(predictors):
    """
        Standardizes predictors (assumes normality)
        Input: predictors xarray object
        Output: predictors xarray object
    """
    for col in predictors.keys():
        #standardize each predictor
        predictors[col] = (('time'), zscore(predictors[col].data))
    return predictors

def stdz_subset(predictors, month_range=month_range):
    """standardizes data for a subset of the year"""
    for col in predictors.keys():
        subset = predictors.sel(time = predictors.time.dt.month.isin(list(month_range)))[col].data
        zscores = (predictors[col] - np.mean(subset)) / np.std(subset)
        predictors[col] = (('time'), zscores)
    return predictors

def stdz_month(predictors):
    """standardizes by month"""
    for month in month_range:
        X_month = predictors.sel(time=predictors.time.dt.month == month)
        for col in predictors.keys():
            subset = X_month.sel(time = slice('1980-01-01', '2005-12-31'))
            mu = float(np.mean(subset[col].data))
            sd = float(np.std(subset[col].data))
            print(mu)
            print(sd)
            X_month[col] = ( ('time'), zscore(X_month[col].data), mu = 0, sd=1)
        if month == list(month_range)[0]:
            X_preds = X_month
        else:
            X_preds = xr.concat([X_preds, X_month], dim = "time")
    return X_preds


def prep_data(obsPath, predictors, lat, lon, dateStart = '1980-01-01', dateEnd = '2014-12-31'):
    """
        Creating readable xarray objects from obs data and predictors
        Input:
            obsPath: filepath entered as a commandline argument
            predictors: existing xarray obj
            lat: latitude; float
            lon: longitude; float
            dateStart (optional): cutoff date for start of data
            dateEnd (optional): cutoff date for end of data
        Output:
            X_all, an xarray object of predictors
            Y_all, an xarray object of obs
    """
    obs = xr.open_dataset(obsPath).sel(time = slice(dateStart, dateEnd))
    X_all = predictors.sel(lat = lat, lon = lon, method = 'nearest').sel(time = slice(dateStart, dateEnd))
    Y_all = obs.sel(lat = lat, lon = lon, method = 'nearest').sel(time = slice(dateStart, dateEnd))
    return X_all, Y_all

def add_month(X_all, Y_all):
    """
        Add month as a predictor in order to make separate models
        Input:
            X_all, an xarray object of predictors
            Y_all, an xarray object of obs
        Output:
            X_all, an xarray object of predictors with a month col
            Y_all, an xarray object of obs with a month col
            all_preds, a list of all keys in the predictors object
    """

    for dataset in [X_all, Y_all]:
        #adding a variable for month into the input and obs datasets
        all_preds = [key for key in dataset.keys()] #the names of the predictors
        dataset['month'] = dataset.time.dt.month
        if not 'month' in all_preds: all_preds += ['month'] #adding "month" to the list of variable names
    return X_all, Y_all, all_preds

def add_constant_col(X_all):
    """
        Adds a column of ones for a constant (since the obs aren't normalized they aren't centered around zero)
        Input:
            X_all, xarray obj of predictors
        Output:
            X_all, xarray obj of predictors
            all_preds, list of predictors
    """
    all_preds = [key for key in X_all.keys()] #the names of the predictors
    X_all['constant'] = 1 + 0*X_all[all_preds[0]] #added the last part so it's dependent on lat, lon, and time
    if not 'constant' in all_preds:
        all_preds += ['constant'] #adding "constant" to the list of variable names
    return X_all, all_preds

def evenOdd(ds):
    """
        Separate even years for training and odd for testing
        Input: xarray dataset
        Output: even and odd year datasets as xarray objects
    """
    ds['timecopy'] = ds['time'] #backup time
    #classify years as even or odd
    ds['time'] =  pd.DatetimeIndex(ds.time.values).year%2 == 0
    even, odd = ds.sel(time = True), ds.sel(time = False)
    #reset to original time data
    even['time'], odd['time'] = even['timecopy'],odd['timecopy']
    return even.drop('timecopy'), odd.drop('timecopy')

def add_month_filter(data):
    """
        Set time dim to just the month. Save time as a variable.
        Input: xarray obj
        Output: xarray obj
    """
    data['timecopy'] = data['time']
    data['time'] = data['month']
    return data


#==============================================================================
"""
Functions for fitting regression model.
"""
#==============================================================================


def fit_linear_model(X, y, keys=None):
    """
        Use linear algebra to compute the betas for a multiple linear regression.
        Input: X (predictors) and y (obs) as xarray objects
        Output: a pandas dataframe with the betas (estimated coefficients) for each predictor.
    """
    if not type(X) is np.matrixlib.defmatrix.matrix:
        keys = [key for key in X.keys()]
        X = np.matrix([X[key].values for key in keys]).transpose() #X matrix; rows are days, columns are variables
    XT = X.transpose()
    betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(XT,X)), XT), y)
    b = pd.DataFrame(index = range(1))
    for i in range(len(keys)):
        b[keys[i]] = betas[0,i] #assigning names to each coefficient
    return b

def fit_monthly_linear_models(X_train, Y_train, preds_to_keep, predictand, conditional):
    """
        Fits a linear model for each month of training data
        Input:
            X_train, an xarray obj of predictors
            Y_train, an xarray obj of observed predictand
            preds_to_keep, a list of key values of predictors to include in the regression
        Output:
            coefMatrix, an array of betas from the regression for each month
    """
    for month in month_range:
        #get obs data
        y = Y_train.sel(time = month)[predictand].values #obs values
        if conditional:
            #filter only days with nonzero precip
            y =  y[y > 0]

        #get just subset of predictors
        x_train_subset = np.matrix([X_train.sel(time = month)[key].values for key in preds_to_keep]).transpose()
        if conditional:
            #filter only days with nonzero precip
            x_train_subset = x_train_subset[Y_train.sel(time = month)[predictand].values > 0, ]

        #calculate coefficients for training data
        coefs = fit_linear_model(x_train_subset, y,
                    keys=preds_to_keep).rename(index = {0: 'coefficient'}).transpose()
        #set up beta matrix
        if month == 1:
            coefMatrix = coefs
        #store betas
        coefMatrix[monthsFull[month-1]] = coefs.values
    coefMatrix = coefMatrix.drop('coefficient', axis=1)
    return coefMatrix

def fit_monthly_lasso_models(X_train, Y_train, predictand, conditional):
    """
        LASSO regressor
        Input:
            X_train, an xarray obj of predictors
            Y_train, an xarray obj of observed predictands
            predictand, string with the var being predicted
        Output:
            an array of betas from the regression for each month
    """
    reg = sklearn.linear_model.Lasso(fit_intercept = False, alpha=0.5)

    #getting predictor names
    keys = [key for key in X_train.drop(['month', 'timecopy']).keys()]

    #running LASSO for each month
    for month in month_range:

        #get obs data
        y = Y_train.sel(time = month)[predictand].values #obs values

        #creating numpy matrices
        X_train_np = np.matrix([X_train.sel(time = month)[key].values for key in keys]).transpose()
        #X_test_np = np.matrix([X_test.sel(time = month)[key].values for key in keys]).transpose()
        reg.fit(X_train_np, y)

        lasso_preds = [keys[i] for i in range(len(reg.coef_)) ]
        betas_LASSO = pd.DataFrame(index = lasso_preds,
                                data = [coef for coef in reg.coef_ ], columns = ['January'])
        if month == 1:
            betas_LASSO_list = betas_LASSO

        betas_LASSO_list[monthsFull[month-1]] = betas_LASSO.values

    return betas_LASSO_list

def fit_annual_lasso_model(X_train, Y_train, predictand, conditional=False):
    """ todo """
    reg = sklearn.linear_model.Lasso(fit_intercept = False, alpha=0.5)

    #getting predictor names
    keys = [key for key in X_train.drop(['month', 'timecopy']).keys()]

    #obs
    y = Y_train[predictand].values #obs values

    #creating numpy matrix
    X_train_np = np.matrix([X_train[key].values for key in keys]).transpose()
    reg.fit(X_train_np, y)

    lasso_preds = [keys[i] for i in range(len(reg.coef_)) ]
    betas_LASSO = pd.DataFrame(index = lasso_preds,
                               data = np.array([[coef for coef in reg.coef_ ]]*len(month_range)).transpose(),
                               columns = [monthsFull[i-1] for i in month_range])
    return betas_LASSO

def fit_annual_OLS(X_train, Y_train, preds_to_keep, predictand, conditional):
    y = Y_train[predictand].values #obs values
    if conditional:
        #filter only days with nonzero precip
        y =  y[y > 0]

    #get just subset of predictors
    x_train_subset = np.matrix([X_train[key].values for key in preds_to_keep]).transpose()
    if conditional:
        #filter only days with nonzero precip
        x_train_subset = x_train_subset[Y_train[predictand].values > 0, ]

    #calculate coefficients for training data
    coefs = fit_linear_model(x_train_subset, y,
                keys=preds_to_keep)
    #store betas
    return pd.DataFrame(index = preds_to_keep,
                               data = np.array(list(coefs.values)*len(month_range)).transpose(),
                               columns = [monthsFull[i-1] for i in month_range])


def fit_logistic(X_train, y, predictand):
    """
        Fits a logistic model for conditional regression (ie for precip)
        The default fit is L2 regulator with C = 1.0
        Input:
            X, an xarray obj of predictors
            y, an xarray obj of observed predictands
            predictand, string with the var being predicted
        Output:
            an array of betas from the regression for each month
    """
    #getting predictor names
    keys = [key for key in X_train.drop(['month', 'timecopy']).keys()]
    X = np.matrix([X_train[key].values for key in keys]).transpose()
    y_binary = y[predictand].values > 0

    #fit logistic equation
    glm = LogisticRegression(penalty = 'l2', C=1)
    glm.fit(X, y_binary)
    logit_preds = [keys[i] for i in range(len(glm.coef_[0])) if glm.coef_[0][i] != 0]
    return pd.DataFrame(index = logit_preds, data = [coef for coef in glm.coef_[0] if coef !=0], columns = ['coefficient']), glm

def save_betas(save_path, coefMatrix, lat, lon, predictand, suffix = ""):
    """
        Save betas to disk
        Input:
            save_path, a string containing the desired save location
            coefMatrix, the matrix of betas to save
            lat, latitude (float)
            lon, longitude (float)
        Output:
            None
    """
    ROOT = os.path.join(save_path,'betas')
    try:
        os.mkdir(ROOT)
    except FileExistsError:
        pass
    fp = os.path.join(ROOT, f"betas{suffix}_{predictand}_{lat}_{lon}.txt")
    try:
        os.remove(fp)
    except: pass
    coefMatrix.to_csv(fp)




#==============================================================================
"""
Functions for generating predictions from model.
"""
#==============================================================================

def predict_linear(X_all, betas, preds_to_keep):
    """
        Predict using each monthly model then combine all predicted data.
        Input:
            X_all, an xarray obj with predictors
            betas, an array of coefficients for each month
            preds_to_keep, the predictors to include in the model
        Output:
            an xarray obj containing a 'preds' column
    """
    X_all_cp = X_all
    X_all_cp['time'] = X_all_cp['timecopy'].dt.month
    X_all_hand = [np.matrix([X_all_cp.sel(time = month)[key].values for key in preds_to_keep]).transpose() for month in month_range]
    for month in month_range:
        X_month = X_all.sel(time=month)
        X_month["preds"] = X_month['time'] + X_month['lat']
        X_month["preds"]= ({'time' : 'time'}, np.matmul(X_all_hand[month-list(month_range)[0]], betas[monthsFull[month-1]]))
        X_month['time'] = X_month['timecopy']
        if month == list(month_range)[0]:
            X_preds = X_month
        else:
            X_preds = xr.concat([X_preds, X_month], dim = "time")

    return X_preds.sortby('time')

def predict_conditional(X_all, betas, logit_betas, predictand, glm, preds_to_keep, thresh = 0.5):
    """
        to do
    """
    # if thresh = 'stochastic':
    #     thresh = np.random.normal(len=X_all.shape[0], mean=0.5, var=0.1)

    X_all_cp = X_all
    X_all_cp['time'] = X_all_cp['timecopy'].dt.month
    X_all_hand = [np.matrix([X_all_cp.sel(time = month)[key].values for key in preds_to_keep]).transpose() for month in month_range]
    for month in month_range:
        X_month = X_all.sel(time=month)
        X_month["preds"] = X_month['time'] + X_month['lat']

        #predict yes/no precip
        classifier = glm.predict_proba(X_all_hand[month-1])[:,1] > thresh
        #predict intensity
        intensity = np.matmul(X_all_hand[month-1], betas[monthsFull[month-1]])

        X_month["preds"]= ({'time' : 'time'}, np.multiply(intensity,classifier))
        X_month['time'] = X_month['timecopy']

        if month == 1:
            X_preds = X_month
        else:
            X_preds = xr.concat([X_preds, X_month], dim = "time")

    return X_preds.sortby('time')


def save_preds(save_path, preds, lat, lon, predictand):
    """
        Saves predictions to disk as netcdf
        Input:
            save_path as str
            preds, xarray obj
            lat, latitude (float)
            lon, longitude (float)
        Output:
            None
    """
    ROOT = os.path.join(save_path,'preds')
    try:
        os.mkdir(ROOT)
    except FileExistsError:
        pass
    fp = os.path.join(ROOT, f"finalPreds_{predictand}_{lat}_{lon}.nc")
    try:
        os.remove(fp)
    except: pass
    preds.to_netcdf(fp)

def inflate_variance(mu, variance, preds):
    """
        Adds a stochastic element by sampling from a normal distribution centered at mu with spread sigma.
        Input:
              mu, mean of sampling ditribution for white noise (float)
              variance, variance of sampling distribution for white noise (float)
              preds, the raw predictions as xarray obj
        Output:
             predictions plus the stochastic elements
    """
    sigma = math.sqrt(variance)
    stochast = np.random.normal(mu, sigma, preds.preds.shape[0])
    preds['preds'] = preds.preds + stochast
    return preds

def SE(y, preds):
    """
        Calculates standard error of the regression.
    """
    n = preds.preds.shape[0]
    stdErrors = np.std(y.values - preds.preds.values)
    return stdErrors * np.sqrt((n-1)/(n-2))

def inflate_variance_SDSM(y, preds, c=12):
    """
        Replicating the "variance inflation factor" from SDSM.
        input:
            c (int), the "variance inflation factor" from SDSM.
            When set to 12, the distribution is roughly normal with
            meam = 0, variance = 1.
    """
    stdError = SE(y, preds)
    v = []
    for i in range(preds.preds.shape[0]):
        Ri = sum([random() for i in range(c)])
        v += [stdError * (Ri -(c/2))]
    preds['preds'] = preds.preds + v
    return preds
