## Linear_downscaling

#### Katrina Wheelan
#### July 9, 2020
#### NCAR

This repository contains custom Jupyter notebooks, Python scripts, and batch scripts for linearly downscaling climate data.

**Directories**:
 * *notebooks/* - Jupyter notebooks for interactive/customizable downscaling
   * *Precip_DownScaling.ipynb* - A Jupyter notebook that linearly downscales precipitation data using large scale predictors. The notebook employs a logistic model to predict yes/no precip and then a linear model to predict intensity. It has an optional stochastic component to correct the distribution.

  * *Temp_Downscaling_MonthlyModel.ipynb* - similarly downscales maximum temperature data using large scale predictors. The model controls by month, employing 3 linear methods: (1) manual linear regression, using the same predictors but different coefficients for each month; (2) LASSO linear regress using a single model; (3) LASSO regression conditioned on month. The chosen model also includes an optional stochastic component.

  * *Tmin_Downscaling_MonthlyModel.ipynb* - downscales min temperature data (IN PROGRESS)

  * *Graphs.ipynb* - Shows various metrics to compare different models 

 * *betas/* - a folder to store betas from regressions for re-use

 * *pythonScripts* - A folder to store scripts called by bash scripts
 
 * *bashScripts/* - Will contain batch scripts for submission to the supercomputer

