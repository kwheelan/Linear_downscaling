
#==============================================================================
"""
A module of plotting functions for the downscaled predictions.

Katrina Wheelan
July 2020

# TODO: make Plot class to avoid passing vars back and forth
"""
#==============================================================================

__all__ = ['plot_monthly_avgs', 'plot_hot_days', 'plot_all_seasons', 'save_stats',
            'plot_dists', 'Plot']

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
    obsSeasonAvg = [plotData.obs.sel(time = slice(str(y)+startDate, str(y+winter) + endDate))[plotData.predictand].mean(dim = 'time') for y in range(startYr, endYr) ]
    modelSeasonAvg = dict()
    for model in plotData.models.keys():
        modelSeasonAvg[model] = [plotData.models[model].sel(time = slice(str(y)+startDate, str(y+winter)+endDate)).preds.mean(dim = 'time') for y in range(startYr, endYr) ]

    movingAvg = [sum([obsSeasonAvg[t+i] for i in range(5)])/5 for t in range(len(obsSeasonAvg)-4)]

    plt.plot(range(startYr, endYr), obsSeasonAvg, '-b', label = 'obs')
    for model in plotData.models.keys():
        plt.plot(range(startYr, endYr), modelSeasonAvg[model], label=model)
    plt.plot(range(startYr+2, endYr-2), movingAvg, '-k', label = "5-yr moving average")
    plt.title(title)
    plt.ylabel('Temp (Celcius)')
    plt.xlabel('Year')
    plt.legend()
    plt.savefig(os.path.join(plotData.plot_path, f"{title.replace(' ','')}.png"))
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



def plot_monthly_avgs(plotData):
    """
        Saving a plot of monthly averages for predictand.
        Input: plotData object
        Output: None
    """
    plotData.obs['time'] = plotData.obs['month']
    modelAvgs = dict()
    for model in plotData.models.keys():
        plotData.models[model]['time'] = plotData.models[model]['month']
        modelAvgs[model] = [float(plotData.models[model].sel(time = m).mean(dim = 'time')['preds']) for m in range(1,13)]
    obsAvgs = [float(plotData.obs.sel(time = m).mean(dim = 'time')[plotData.predictand]) for m in range(1,13)]
    plt.plot(monthsAbrev, obsAvgs, '-b', label = 'obs')
    for model in plotData.models.keys():
        plt.plot(monthsAbrev, modelAvgs[model], label=model)
    plt.title("Mean Monthly Max Temperature for Observed and Modeled Data")
    plt.ylabel('Temperature (Celcius)')
    plt.xlabel('Month')
    plt.legend()
    plt.savefig(os.path.join(plotData.plot_path, 'monthly_means.png'))
    plt.clf()

def plot_cond_days(plotData, title, comp = "greater", thresh = 35):
    """
        Saving a plot of days that satisify cond.
        Ex. number of days over 35 degrees C
        Input: plotData, Plot obj
               title - a str plot title and plot file name
               comp - "greater" or "less" for comparison to the threshold
               thresh - cutoff for days to count
        Output: None
    """
# to do: divide by number of years
    plotData.obs['time'] = plotData.obs.month
    for key in plotData.models.keys():
        plotData.models[key]['time'] = plotData.models[key].month
    if comp.lower() == "greater":
        obsDaysCount = [sum(plotData.obs.sel(time=m)[plotData.predictand].values > thresh) for m in range(1,13)]
        modelDaysCount = dict()
        for model in plotData.models.keys():
            modelDaysCount[model] = [sum(plotData.models[model].sel(time=m).preds.values > thresh) for m in range(1,13)]
    else:
        obsDaysCount = [sum(plotData.obs.sel(time=m)[plotData.predictand].values < thresh) for m in range(1,13)]
        modelDaysCount = dict()
        for model in plotData.models.keys():
            modelDaysCount[model] = [sum(plotData.models[model].sel(time=m).preds.values > thresh) for m in range(1,13)]
    plt.plot(monthsAbrev, obsDaysCount, '-b', label = 'obs')
    for model in plotData.models.keys():
        plt.plot(monthsAbrev, modelDaysCount[model], label=model)
    plt.title(title)
    plt.ylabel('Number of Days')
    plt.xlabel('Month')
    plt.legend()
    plt.savefig(os.path.join(plotData.plot_path, f"{title.replace(' ','')}.png"))
    plt.clf()

def plot_hot_days(plotData):
        """
            Saving a plot of days over 35 degrees C
            Input: plotData.obs (obs data) as xarray obj
                   preds as xarray obj
                   save_path, str of location for saving img
                   lat - latitude as a float
                   lon - longitude as a float
            Output: None
        """
        plot_cond_days(plotData, title="Number of Days over 35 Degrees Celcius", comp="greater", thresh=35)

def plot_cold_days(plotData):
    """ todo """
    plot_cond_days(plotData, title="Number of Days under 0 Degrees Celcius", comp="less", thresh = 0)

def plot_dist(plotData, data, title):
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
    plt.hist(eval(data), bins = 25)
    plt.title(title)
    plt.ylabel("Number of points") # TODO: make into percentage
    plt.xlabel(plotData.predictand)
    plt.savefig(os.path.join(plotData.plot_path, f"{title.replace(' ','')}.png"))
    plt.clf()

def plot_dists(plotData):
    """
        Plots obs and model distribution.
    """
    plot_dist(plotData, "plotData.obs[plotData.predictand]", "Distribution of Observed Data")
    for model in plotData.models.keys():
        plot_dist(plotData, f"plotData.models['{model}'].preds", f"Distribution of {model} Predictions")

#===============================================================================
"""
Saving summary statistics.
"""
#===============================================================================

def save_stats(plotData):
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
    f = open(os.path.join(plotData.plot_path, "stats.txt"), "w")
    f.write("Observations:\n")
    f.write(f"Mean: {float(np.mean(plotData.obs[plotData.predictand]))}\n")
    f.write(f"Variance: {float(np.var(plotData.obs[plotData.predictand]))}\n\n")
    for key in plotData.models.keys():
        f.write(f"Modeled Data ({key}):\n")
        f.write(f"Mean: {float(np.mean(plotData.models[key].preds))}\n")
        f.write(f"Variance: {float(np.var(plotData.models[key].preds))}\n")
    f.close()
