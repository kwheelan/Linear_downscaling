from regression_methods import *
x = load_all_predictors()

import numpy as np

s, n = slice(32, 35), slice(36, 39)
w, e = slice(-102, -99), slice(-98, -93)

AprSep, OctMar = [4, 5, 6, 7, 8, 9], [10, 11, 12, 1, 2, 3]

for lat in 's','n':
    for lon in 'e','w':
        for seas in 'AprSep', 'OctMar':
            title = f"{lat}{lon}_{seas}.txt"
            y = x.sel(lat = eval(lat), lon=eval(lon), time=x.time.dt.month.isin(eval(seas)))
            np.savetxt(title, y.to_dataframe().corr(), delimiter= ',')
