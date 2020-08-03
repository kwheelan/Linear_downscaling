
#==============================================================================
"""
A module of plotting functions for the downscaled predictions.

Katrina Wheelan
July 2020

# TODO: make Plot class to avoid passing vars back and forth
"""
#==============================================================================

__all__ = ['plot_monthly_avgs', 'plot_hot_days', 'plot_all_seasons', 'save_stats',
            'plot_dist', 'Plot']

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

#==============================================================================
"""Creating a Plot class to minimize data-passing"""
#==============================================================================

class Plot:
    """Stores appropriate data"""

    __slots__ = ['plot_path', 'lat', 'lon', 'predictand', 'obs', 'models']

    def __init__(self, save_path, lat, lon, predictand, obs, models):
        folder = f"plots_lat{lat}_lon{lon}"
        try:
            os.mkdir(os.path.join(save_path, folder))
        except:
            pass #assume folder exists
        self.plot_path =  os.path.join(save_path, folder)
        self.lat, self.lon = lat, lon
        self.predictand = predictand
        self.obs = obs #Y_all xarray obs
        self.models = models #an array of xarray objs; must have preds as a var


#===============================================================================
"""
Plotting functions for downscaled data.
"""
#===============================================================================

def annualSeasonPlot(plotData, startDate, endDate, title, startYr = 1980, endYr = 2014):
    """
        Plots seasonal avgs compared to an observed moving average
        Input:
            plotData, a Plot object
            startDate of season as a string; ex. '12-01'
            endDate of season as a string; ex. '02-28'
            title as a string; ex. 'Season plot: winter'
            startYr, optional start year of data
            endYr, optional end year of data
        Output: None
    """
    winter = startDate[0:2] > endDate[0:2]
    startDate, endDate = '-' + startDate, '-' + endDate
    obsSeasonAvg = [plotData.obs.sel(time = slice(str(y)+startDate, str(y+winter) + endDate))[predictand].mean(dim = 'time') for y in range(startYr, endYr) ]
    for model in plotData.models:
        modelSeasonAvg = dict()
        modelSeasonAvg[model] = [model.sel(time = slice(str(y)+startDate, str(y+winter)+endDate)).preds.mean(dim = 'time') for y in range(startYr, endYr) ]

    movingAvg = [sum([obsSeasonAvg[t+i] for i in range(5)])/5 for t in range(len(obsSeasonAvg)-4)]

    plt.plot(range(startYr, endYr), obsSeasonAvg, '-b', label = 'obs')
    for model in plotData.models:
        plt.plot(range(startYr, endYr), modelSeasonAvg[model], label=model)
    plt.plot(range(startYr, endYr), movingAvg, '-k', label = "5-yr moving average")
    plt.title(title)
    plt.ylabel('Temp (Celcius)')
    plt.xlabel('Year')
    plt.legend()
    plot_path = make_plot_folder(save_path, lat, lon)
    plt.savefig(os.path.join(plot_path, f"{title.replace(' ','')}.png"))
    plt.clf()


def plot_all_seasons(plotData):
    """
        Plots 4 plots of seasonal avgs compared to an observed moving average
        Input:
            plotData.obs (obs data) as xarray obj
            preds as xarray obj
            save_path, str of location for saving img
            lat - latitude as a float
            lon - longitude as a float
        Output: None
    """
    plotData.obs['time'] = plotData.obs['timecopy']
    annualSeasonPlot(plotData, '12-01', '02-28', "Seasonal Plot Dec-Jan-Feb")
    annualSeasonPlot(plotData, '03-01', '05-30', "Seasonal Plot Mar-Apr-May")
    annualSeasonPlot(plotData, '06-01', '08-31', "Seasonal Plot Jun-Jul-Aug")
    annualSeasonPlot(plotData, '09-01', '11-30', "Seasonal Plot Sep-Oct-Nov")



def plot_monthly_avgs(plotData.obs, preds, save_path, lat, lon, predictand):
    """
        Saving a plot of monthly averages for predictand.
        Input: plotData.obs (obs data) as xarray obj
               preds as xarray obj
               save_path, str of location for saving img
               lat - latitude as a float
               lon - longitude as a float
        Output: None
    """
    plotData.obs['time'], preds['time'] = plotData.obs['month'], preds['month']
    modelAvgs = [float(preds.sel(time = m).mean(dim = 'time')['preds']) for m in range(1,13)]
    obsAvgs = [float(plotData.obs.sel(time = m).mean(dim = 'time')[predictand]) for m in range(1,13)]
    plt.plot(monthsAbrev, obsAvgs, '-b', label = 'obs')
    plt.plot(monthsAbrev, modelAvgs, '-r', label='model')
    plt.title("Mean Monthly Max Temperature for Observed and Modeled Data")
    plt.ylabel('Temperature (Celcius)')
    plt.xlabel('Month')
    plt.legend()
    plot_path = make_plot_folder(save_path, lat, lon)
    plt.savefig(os.path.join(plot_path, 'monthly_means.png'))
    plt.clf()

def plot_cond_days(plotData.obs, preds, save_path, lat, lon, predictand, title = "Conditional Day Count", comp = "greater", thresh = 35):
    """
        Saving a plot of days that satisify cond.
        Ex. number of days over 35 degrees C
        Input: plotData.obs (obs data) as xarray obj
               preds as xarray obj
               save_path, str of location for saving img
               cond - str condition (ex. "> 35")
               lat - latitude as a float
               lon - longitude as a float
               title - a str plot title and plot file name
        Output: None
    """
# to do: divide by number of years
    plotData.obs['time'], preds['time'] = plotData.obs.month, preds.month
    if comp.lower() == "greater":
        obsDaysCount = [sum(plotData.obs.sel(time=m)[predictand].values > thresh) for m in range(1,13)]
        modelDaysCount = [sum(preds.sel(time=m).preds.values > thresh) for m in range(1,13)]
    else:
        obsDaysCount = [sum(plotData.obs.sel(time=m)[predictand].values < thresh) for m in range(1,13)]
        modelDaysCount = [sum(preds.sel(time=m).preds.values < thresh) for m in range(1,13)]
    plt.plot(monthsAbrev, obsDaysCount, '-b', label = 'obs')
    plt.plot(monthsAbrev, modelDaysCount, '-r', label='model')
    plt.title(title)
    plt.ylabel('Number of Days')
    plt.xlabel('Month')
    plt.legend()
    plot_path = make_plot_folder(save_path, lat, lon)
    plt.savefig(os.path.join(plot_path, f"{title.replace(' ','')}.png"))
    plt.clf()

def plot_hot_days(plotData.obs, preds, save_path, lat, lon):
        """
            Saving a plot of days over 35 degrees C
            Input: plotData.obs (obs data) as xarray obj
                   preds as xarray obj
                   save_path, str of location for saving img
                   lat - latitude as a float
                   lon - longitude as a float
            Output: None
        """
        plot_cond_days(plotData.obs, preds, save_path, lat, lon, predictand = 'tmax', title="Number of Days over 35 Degrees Celcius", comp="greater", thresh=35)

def plot_dist(data, title, save_path, lat, lon, predictand):
    """
        Generates and saves a plot of the distribution of given data
        Input:
            data as xarray obj
            title, a string to appear on plot
            save_path, str of location for saving img
            lat - latitude as a float
            lon - longitude as a float
        Output:
            none
    """
    plt.hist(data, bins = 25)
    plt.title(title)
    plt.ylabel("Number of points") # TODO: make into percentage
    plt.xlabel(predictand)
    plot_path = make_plot_folder(save_path, lat, lon)
    plt.savefig(os.path.join(plot_path, f"{title.replace(' ','')}.png"))
    plt.clf()

#===============================================================================
"""
Saving summary statistics.
"""
#===============================================================================

def save_stats(plotData.obs, preds, lat, lon, save_path, predictand):
    """
        Saving a txt file with data on the modeled predictions
        Input: plotData.obs (obs data) as xarray obj
               preds as xarray obj
               save_path, str of location for saving img
               lat - latitude as a float
               lon - longitude as a float
        Output: None
        # TODO: r-squared; standard error
    """
    plot_path = make_plot_folder(save_path, lat, lon)
    f = open(os.path.join(plot_path, "stats.txt"), "w")
    f.write("Observations:\n")
    f.write(f"Mean: {float(np.mean(plotData.obs[predictand]))}\n")
    f.write(f"Variance: {float(np.var(plotData.obs.tmax))}\n\n")
    f.write("Modeled Data:\n")
    f.write(f"Mean: {float(np.mean(preds.preds))}\n")
    f.write(f"Variance: {float(np.var(preds.preds))}\n")
    f.close()
