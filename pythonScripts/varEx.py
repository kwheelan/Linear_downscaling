from regression_methods import *
x = load_all_predictors().sel(time = slice('1980-01-01','2005-12-31'))
import xarray as xr

obs_file = "/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc"
obs = xr.open_dataset(obs_file).sel(time = slice('1980-01-01','2005-12-31'))

import numpy as np
import pandas as pd

s, n = slice(32, 35), slice(36, 39)
w, e = slice(-102, -99), slice(-98, -93)

coord_dict = {'n': [36.125, 38.125], 's':[32.125, 34.125], 'w': [-101.875, -99.875],'e' : [-97.875, -95.875, -93.875]}

monthsFull = ['January','February', 'March','April','May','June','July','August','September','October','November','December']

all_preds = [key for key in x.keys()] #the names of the predictors

def var_ex(x, y_full):
    for month in range(1, 13):
        #get obs data
        y = y_full.sel(time = y_full.time.dt.month == month)['tmax'].values
        matrix = pd.DataFrame([x.sel(time = x.time.dt.month == month)[key].values.flatten() for key in all_preds] + [y.flatten()])
        corrs = matrix.transpose().corr()[len(all_preds)][0:-1]
    
        if month == 1:
            #create dataframe
            exVar = pd.DataFrame(index = range(1))
            for i in range(len(all_preds)):
                exVar[all_preds[i]] = corrs[i] #assigning names to each coefficient
            exVar = exVar.rename(index = {0: 'January'}).transpose() 
        else:
            #otherwise, add the data as a column
            exVar[monthsFull[month-1]] = corrs.values
        
    exVar = round(exVar**2, 3) #square to get R^2 and round to look nicer
    return exVar

for lat in 's','n':
    for lon in 'w','e':
        title = f"{lat}{lon}.txt"
        data = x.sel(lat = eval(lat), lon=eval(lon))
        y = obs.isel(lat = obs.lat.isin(coord_dict[lat]), lon=obs.lon.isin(coord_dict[lon]))
        np.savetxt(title, var_ex(data, y), delimiter = ',')
