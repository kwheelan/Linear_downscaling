# Linear_downscaling

This repository contains two Jupyter notebooks for linear downscaling.

*Precip_DownScaling.ipynb* linearly downscales precipitation data using large scale predictors. The notebook employs a logistic model to predict yes/no precip and then a linear model to predict intensity. It has an optional stochastic component to correct the distribution.

*Temp_Downscaling_MonthlyModel.ipynb* similarly downscales maximum temperature data using large scale predictors. There is only a linear model. Eventually the model will condition on month. The model also includes an optional stochastic component.

*Tmin_Downscaling_MonthlyModel.ipynb* downscales min temperature data (IN PROGRESS)

*Graphs.ipynb* Shows various metrics to compare different models 

*betas/* a folder to store betas from regressions for re-use

*pythonScripts* A folder to store scripts called by bash scripts
