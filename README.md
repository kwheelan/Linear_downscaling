## Linear_downscaling

#### Katrina Wheelan
#### July 28, 2020
#### NCAR

(IN PROGRESS)

This repository contains custom Jupyter notebooks, Python scripts, and batch scripts for linearly downscaling climate data.

**Directories**:
 * *notebooks/* - Jupyter notebooks for interactive/customizable downscaling
   * *Precip_DownScaling.ipynb* - A Jupyter notebook that linearly downscales precipitation data using large scale predictors. The notebook employs a logistic model to predict yes/no precip and then a linear model to predict intensity. It has an optional stochastic component to correct the distribution.

  * *Temp_Downscaling_MonthlyModel.ipynb* - similarly downscales maximum temperature data using large scale predictors. The model controls by month, employing 3 linear methods: (1) manual linear regression, using the same predictors but different coefficients for each month; (2) LASSO linear regress using a single model; (3) LASSO regression conditioned on month. The chosen model also includes an optional stochastic component (an approximation of the "variance inflation" for SDSM).

  * *Tmin_Downscaling_MonthlyModel.ipynb* - downscales min temperature data 

  * *Graphs.ipynb* - Shows various metrics to compare different models 

 * *betas/* - a folder to store betas from regressions for re-use

 * *pythonScripts* - A folder to store scripts called by bash scripts
    * *regress.py* - Optimizable python file to do regression for downscaling
    * *regression_tools.py* - A module of functions to perform the necessary steps for downscaling
    * *plotting.py* - A module of plotting functions for the modeled data
 
 * *bashScripts/* - Contains batch scripts for submission to the NCAR supercomputer

