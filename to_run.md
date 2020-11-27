## Linear_downscaling

#### Katrina Wheelan
#### November, 2020
#### NCAR

## Instructions to run the scripts:

The bash script to run on the supercomputer is *bashScript/regress_OLS.sh*.
In the script, you can edit the USER, the filepath for the observations (OBS),
the save location (LOCATION), and the lat-lon points for the calculations.


*pythonScripts/regress.py* is the Python script that runs the actual code. It calls on two modules
of functions: *regression_methods.py* and *plotting.py*. You shouldn't have to
edit any of these three files.


*pythonScripts/settings.txt* is a text file containing the options for the model.
Specifically:
  - lat = latitude if not specified in command line (if run from bash file, ignore)
  - lon = longitude if not specified in command line (if run from bash file, ignore)
  - predictand = 'tmax', 'tmin', or 'prec'
  - preds_surface = list of surface predictors
  - preds_level = list of predictors and their levels
  - dateStart = starting date of data
  - dateEnd = ending date of data
  - obs_path = filepath of observations
  - save_path = save location of predictions
  - method = 'OLS' or 'LASSO'
  - monthly = True if run separate models for each month
  - train = True if separate train and testing data
  - transform = True if fourth root transformation (for prec)
  - stdize = True if standardize predictors
  - stdize_y = True if standardize predictand
  - conditional = True if do a two-step regression for conditional process (prec)
  - stochastic_thresh = True if stochastic threshold for yes/no precip
  - static_thresh = value for yes/no precip cutoff
  - inflate = True if add noise to predictions
  - inflate_var = magnitude of variance inflation; should match SDSM value roughly
