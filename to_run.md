## Linear_downscaling

#### Katrina Wheelan
#### February, 2021
#### NCAR

## Instructions to run the scripts:

*pythonScripts/regress.py* is the Python script that runs the actual code. It calls on two modules
of functions: *regression_methods.py* and *plotting.py*. *gen_time_series.py* is a similar script to 
*regress.py*, and it generates a time series and plots from a GCM and existing regression model. You shouldn't have to
edit any of these files.

#### Bash scripts

The bash scripts to run on the supercomputer are in the *bashScripts* folder.

*regress_OLS.sh* runs the regression and creates a time series and plots.
In the script, you can edit the USER, the filepath for the observations (OBS),
the save location (LOCATION), and the lat-lon points for the calculations.
Make sure that the fifth line of code that starts with "cd" navigates to the
filepath where the *regress.py* script is located.

*gen_time_series.sh* runs the regression on a GCM from an existing regression model. 
cd to where *regress.py* is saved. Uncomment either the first or the second block of four lines
depending if the GCM data is historical (first block) or future (second block). Similarly,
set the GCM by uncommenting either the first or second two lines under the "set GCM comment."
LAT and LON should be set to the appropriate coordinate. BETAS is where the model betas folder is
saved (this folder should contain both the betas and the metadata). ROOT and SERIES correspond to
the filepath of the GCM and should not need to be changed.

*regress_series.sh* should do everything at once, going through the entire lat-lon grid, 
running a regression model, and using these betas to create a time series and plots for
the historical and future data for a specified GCM. You shouldn't need to alter much here.
HOME is where the python scripts are saved. You can toggle the lines under "set GCM" to 
set the GCM. OBS is the OBS filepath. You can also customize where the settings.txt should be
read in from by changing SETTINGS. 

#### Settings.txt variables

*pythonScripts/settings.txt* is a text file containing the options for the model.
Specifically:
  - predictand = 'tmax', 'tmin', or 'prec'
  - preds_surface = list of surface predictors
  - preds_level = list of predictors and their levels

  - dateStart = starting date of data
  - dateEnd = ending date of data

  - obs_path = filepath of observations
  - save_path = save location of predictions

  - method = 'OLS' or 'LASSO'
  - monthly = True if run separate models for each month
  - apr_sep = True if the model should be restricted to just April through September data

  - train = True if separate train and testing data
  - transform = True if fourth root transformation (for prec)
  - stdize = True if standardize predictors
  - stdize_y = True if standardize predictand

  - conditional = True if do a two-step regression for conditional process (prec)
  - stochastic_thresh = True if stochastic threshold for yes/no precip
  - static_thresh = value for yes/no precip cutoff

  - inflate = True if add noise to predictions
  - inflate_var = variance of the normal distribution from which to sample the noise
  
  #### Output file structure
  
  If you run regress_series.sh, the output should be saved in /glade/work/[USERNAME]/Linear_downscaling/output/
  
  Inside this folder, you'll have folders for ERA-I and any GCM. 
  Inside each will be folders for historical and future time periods. 
  Inside the time folders will be folders for each lat-lon point and predictand.
  These folders contain:
   - *betas/* with betas.txt, metadata.txt, and possibly logit_betas.txt (if logistic)
   - *plots/* 
       - *distributionPlots/*
       - *seasonalPlots/* (for only historical data)
       - *timeSeriesPlots/*
       - *summary.txt*, summary statistics for the model
   - *timeseries/* containing a netCDF of the predictions
