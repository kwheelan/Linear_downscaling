"""
A file to linearly downscale predictors

Right now this just does linear regression and uses all predictors that are passed.
Updated 6.23.2020, K. Wheelan

Usage: regress.py <lat> <lon> <obs filepath> <location to save data> <any pred file paths>+

 """

#TODO:
#add a switch for Lasso
#add stochasticity
#add a switch for two step regression (precip)

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
lat = float(sys.argv[1])
lon = float(sys.argv[2])
obsPath = sys.argv[3] #filepath for obs data
save_path = sys.argv[4]
preds = [ path for path in sys.argv[5:] ] #paths for preds files



### TODO:
#update this based on given predictors
ROOT = '/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/' #where the files are saved
EXT = '_19790101-20181231_dayavg_mpigrid.nc' #date range at the end of the file
SERIES = 'ERAI_NAmerica'

#The variables to use
surface_predictors = ['mslp', 'uas', 'vas'] #, 'ps']
#Each of these predictors is taken at each pressure level below
other_predictors = ['Q', 'RH', 'U', 'V', 'Z', 'Vort', 'Div']
levels = [500, 700, 850] #pressure levels

#Surface predictors
for var in surface_predictors:
    file = ROOT + var + '_' + SERIES + '_surf' + EXT
    if var == 'mslp': predictors = xr.open_dataset(file)[var]
    predictors = xr.merge([predictors, xr.open_dataset(file)[var]])

#Other predictors (at multiple pressure levels)
for var in other_predictors:
    for level in levels:
        file = ROOT + var + '_' + SERIES + '_p' + str(level) + EXT
        predictors = xr.merge([predictors, xr.open_dataset(file).rename({var: var + '_p' + str(level)})])

#convert to pandas dataframe in order to use scikit-learn later

def standardize(variable):
    """Standardizing a variable (converting to z-scores)
    Takes as input an Xarray dataset; outputs a standardized Xarray object"""
    return (variable - np.mean(variable)) / np.std(variable)

for col in [i for i in predictors.keys()]:
    #standardize each predictor
    predictors[col] = standardize(predictors[col])


#get observational data
obs = xr.open_mfdataset(obsPath).sel(time = slice('1979-01-01','2014-12-31'))
#getting training data
#slicing input data to the date range of obs (through 2014)
X_all = predictors.sel(lat = lat, lon = lon, method = 'nearest').sel(time = slice('1980-01-01','2014-12-31'))
Y_all = obs.sel(lat = lat, lon = lon, method = 'nearest').sel(time = slice('1980-01-01','2014-12-31'))


#condition by month
monthsAbrev = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthsFull = ['January','February', 'March','April','May','June','July','August','September','October','November','December']

for dataset in [X_all, Y_all]:
    #adding a variable for month into the input and obs datasets
    all_preds = [key for key in dataset.keys()] #the names of the predictors
    dataset['month'] = dataset.time.dt.month
    if not 'month' in all_preds: all_preds += ['month'] #adding "month" to the list of variable names

#adding a column of ones for a constant (since the obs aren't normalized they aren't centered around zero)
all_preds = [key for key in X_all.keys()] #the names of the predictors
X_all['constant'] = 1 + 0*X_all[all_preds[0]] #added the last part so it's dependent on lat, lon, and time
if not 'constant' in all_preds: all_preds += ['constant'] #adding "constant" to the list of variable names

#separate even years for training and odd for testing
def evenOdd(ds):
    """Input: xarray dataset
        Output: even and odd year datasets as xarray objects"""
    ds['time-copy'] = ds['time']
    #classify years as even or odd
    ds['time'] =  pd.DatetimeIndex(ds.time.values).year%2 == 0
    even, odd = ds.sel(time = True), ds.sel(time = False)
    even['time'], odd['time'] = even['time-copy'],odd['time-copy']
    return even.drop('time-copy'), odd.drop('time-copy')

X_train, X_test = evenOdd(X_all)
Y_train, Y_test = evenOdd(Y_all)

#creating a month filter
for data in [X_train, X_test, Y_train, Y_test]:
    data['timecopy'] = data['time']
    data['time'] = data['month']

## TODO:
#saving training/testing data

# Fit regression model
#TODO verify this chunk
preds_to_drop = ["month", "lat", "lon"]
preds_to_keep = [x for x in all_preds if not x in preds_to_drop]

def fit_linear_model(X, y, keys=None):
    """Use linear algebra to compute the betas for a multiple linear regression.
       Input: X (predictors) and y (obs) as xarray objects
       Output: a pandas dataframe with the betas (estimated coefficients) for each predictor"""
    if not type(X) is np.matrixlib.defmatrix.matrix:
        keys = [key for key in X.keys()]
        X = np.matrix([X[key].values for key in keys]).transpose() #X matrix; rows are days, columns are variables
    XT = X.transpose() #X transpose
    betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(XT,X)), XT), y)
    b = pd.DataFrame(index = range(1))
    for i in range(len(keys)):
        b[keys[i]] = betas[0,i] #assigning names to each coefficient
    return b

#fit a different model for each month

for month in range(1, 13):
    #get obs data
    y = Y_train.sel(time = month).tmax.values #obs values

    #get just subset of predictors
    x_train_subset = np.matrix([X_train.sel(time = month)[key].values for key in preds_to_keep]).transpose()
    x_test_subset = np.matrix([X_test.sel(time = month)[key].values for key in preds_to_keep]).transpose()

    #calculate coefficients for training data
    coefs = fit_linear_model(x_train_subset, y,
                keys=preds_to_keep).rename(index = {0: 'coefficient'}).transpose()

    if month == 1:
        coefMatrix = coefs

    coefMatrix[monthsFull[month-1]] = coefs.values

coefMatrix = coefMatrix.drop('coefficient', axis=1)

#get linear hand-selected predictor test data
x_test_subset = [np.matrix([X_test.sel(time = month)[key].values for key in preds_to_keep]).transpose() for month in range(1,13)]


#saves the betas
ROOT = os.path.join(save_path,'betas')
try:
    os.mkdir(ROOT)
except FileExistsError:
    pass
betas = "coefMatrix"
fp = os.path.join(ROOT, '{}_tmax_{}_{}.nc'.format(betas, str(lat),str(lon)))
try:
    os.remove(fp)
except: pass
eval(betas).to_csv(fp)

#run model to predict for even AND odd years
X_all_cp = X_all
X_all_cp['time'] = X_all_cp['time-copy'].dt.month
x_all_hand = [np.matrix([X_all_cp.sel(time = month)[key].values for key in preds_to_keep]).transpose() for month in range(1,13)]

def knit_data(x_all, betas):
    for month in range(1,13):
        X_month = X_all_cp.sel(time=month)
        X_month["preds"] = X_month['time'] + X_month['lat']
        X_month["preds"]= ({'time' : 'time'}, pd.DataFrame(np.matmul(x_all[month-1], betas[monthsFull[month-1]])).values[0])
        X_month['time'] = X_month['time-copy']
        if month == 1:
            X_preds = 0
            X_preds = X_month
        else:
            X_preds = xr.concat([X_preds, X_month], dim = "time")

    return X_preds.sortby('time')

#predictions
X_preds_lin = knit_data(x_all_hand, coefMatrix)

#saves the fnal predictions (no stochastic component) as netCDF
ROOT = os.path.join(save_path,'preds')
try:
    os.mkdir(ROOT)
except FileExistsError:
    pass
model = "lin"
fp = '/glade/work/kwheelan/linear_data/finalPreds_{}_tmax_{}_{}.nc'.format(model, str(lat),str(lon))
try:
    os.remove(fp)
except: pass
eval("X_preds_" + model).to_netcdf(fp)
