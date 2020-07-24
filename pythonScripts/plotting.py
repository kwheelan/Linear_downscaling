"""
A module of plotting functions for the downscaled predictions.
"""

__all__ = ['plot_monthly_avgs', 'plot_hot_days', 'plot_all_seasons']

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

def plot_cond_days(Y_all, preds, cond, save_path, lat, lon, title = "Conditional Day Count"):
    """
        Saving a plot of days that satisify cond.
        Ex. number of days over 35 degrees C
        Input: Y_all (obs data) as xarray obj
               preds as xarray obj
               save_path, str of location for saving img
               cond - str condition (ex. "> 35")
               lat - latitude as a float
               lon - longitude as a float
               title - a str plot title and plot file name
        Output: None
    """
    Y_all['time'], preds['time'] = Y_all.month, preds.month
    obsDaysCount = [sum(Y_all.sel(time=m).tmax.values eval(cond)) for m in range(1,13)]
    modelDaysCount = [sum(lin.sel(time=m).preds.values eval(cond)) for m in range(1,13)]
    plt.plot(monthsAbrev, obsAvgs, '-b', label = 'obs')
    plt.plot(monthsAbrev, modelAvgs, '-r', label='model')
    plt.title(title)
    plt.ylabel('Temperature (Celcius)')
    plt.xlabel('Month')
    plt.legend()
    plt.show()
    plot_path = make_plot_folder(save_path, lat, lon)
    plt.saveimg(os.path.join(plot_path, f"{title.replace(' ','')}.png"))

def plot_hot_days(Y_all, preds, save_path, lat, lon):
        """
            Saving a plot of days over 35 degrees C
            Input: Y_all (obs data) as xarray obj
                   preds as xarray obj
                   save_path, str of location for saving img
                   lat - latitude as a float
                   lon - longitude as a float
            Output: None
        """
        plot_cond_days(Y_all, preds, cond = "> 35", save_path, lat, lon, title="Number of Days over 35 Degrees Celcius")

def annualSeasonPlot(Y_all, preds, startDate, endDate, title = "Seasonal Plot"):
    """
        Plots seasonal avgs compared to an observed moving average
        Input:
            Y_all (obs data) as xarray obj
            preds as xarray obj
            startDate of season as a string; ex. '12-01'
            endDate of season as a string; ex. '02-28'
            title as a string; ex. 'Season plot: winter'
        Output: None
    """
    winter = startDate[0:2] > endDate[0:2]
    startDate, endDate = '-' + startDate, '-' + endDate
    obsSeasonAvg = [Y_all.sel(time = slice(str(y)+startDate, str(y+winter) + endDate)).tmax.mean(dim = 'time') for y in range(1980, 2014) ]
    modelSeasonAvg = [preds.sel(time = slice(str(y)+startDate, str(y+winter)+endDate)).preds.mean(dim = 'time') for y in range(1980, 2014) ]
    movingAvg = [sum([obsSeasonAvg[t+i] for i in range(5)])/5 for t in range(len(obsSeasonAvg)-4)]

    plotModels(range(1980, 2014), [obsSeasonAvg, modelSeasonAvg])
    plt.plot(range(1983, 2013), movingAvg, '-k', label = "5-yr moving average")
    plt.title(title)
    plt.ylabel('Temp (Celcius)')
    plt.xlabel('Year')
    plt.legend()
    plt.show()
    plt.saveimg(os.path.join(plot_path, f"{title.replace(' ','')}.png"))

def plot_all_seasons(Y_all, preds):
    """
        Plots 4 plots of seasonal avgs compared to an observed moving average
        Input:
            Y_all (obs data) as xarray obj
            preds as xarray obj
        Output: None
    """
    annualSeasonPlot('12-01', '02-28', "Seasonal Plot: Dec-Jan-Feb")
    annualSeasonPlot('03-01', '05-30', "Seasonal Plot: Mar-Apr-May")
    annualSeasonPlot('06-01', '08-31', "Seasonal Plot: Jun-Jul-Aug")
    annualSeasonPlot('09-01', '11-30', "Seasonal Plot: Sep-Oct-Nov")
