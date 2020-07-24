"""
A module of plotting functions for the downscaled predictions.
"""

__all__ = ['plot_monthly_avgs']

#import dependencies
import xarray as xr
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import os

#Set globals
monthsAbrev = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthsFull = ['January','February', 'March','April','May','June','July','August','September','October','November','December']


def make_plot_folder(save_path, lat, lon):
    """
        Makes and returns the filepath for a coordinate specific plot folder.
        Input:
            save_path - str of save location filepath
            lat - latitude as a float
            lon - longitude as a float
        Output:
            the filepath of the newly created folder as a string
    """
    folder = f"plots_lat{lat}_lon{lon}"
    try:
        os.mkdir(os.path.join(save_path, folder))
    except:
        pass #assume folder exists
    return folder

def plot_monthly_avgs(Y_all, preds, save_path, lat, lon):
    """
        Saving a plot of monthly averages for predictand.
        Input: Y_all (obs data) as xarray obj
               preds as xarray obj
               save_path, str of location for saving img
               lat - latitude as a float
               lon - longitude as a float
        Output: None
    """
    Y_all['time'], preds['time'] = Y_all.month, preds.month
    modelAvgs = [float(preds.sel(time = m).mean(dim = 'time').preds) for m in range(1,13)]
    obsAvgs = [float(Y_all.sel(time = m).mean(dim = 'time').tmax) for m in range(1,13)]
    plt.plot(monthsAbrev, obsAvgs, '-b', label = 'obs')
    plt.plot(monthsAbrev, modelAvgs, '-r', label='model')
    plt.title("Mean Monthly Max Temperature for Observed and Modeled Data")
    plt.ylabel('Temperature (Celcius)')
    plt.xlabel('Month')
    plt.legend()
    plt.show()
    plot_path = make_plot_folder(save_path, lat, lon)
    plt.saveimg(os.path.join(plot_path, 'monthly_means.png'))
