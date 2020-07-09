# Linear_downscaling

This repository contains two Jupyter notebooks for linear downscaling.

*Linear_DownScaling.ipynb* linearly downscales precipitation data using large scale predictors. The notebook employs a logistic model to predict yes/no precip and then a linear model to predict intensity. It has an optional stochastic component to correct the distribution.

*Temp_Downscaling.ipynb* (NOT COMPLETE) similarly downscales maximum temperature data using large scale predictors. There is only a linear model. Eventually the model will condition on month. The model also includes an optional stochastic component.
