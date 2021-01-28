

#==============================================================================
"""
A module of plotting functions for the downscaled predictions.

Katrina Wheelan
August 2020

"""
#==============================================================================

__all__ = ['Plot', 'plot_monthly_avgs', 'plot_hot_days', 'plot_cold_days', 'plot_all_seasons',
            'plot_annual_avgs', 'plot_annual_avgs_bar', 'save_stats', 'plot_dists',
            'boxplot', 'violin', 'plot_all', 'plot_all_future']

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

titleDict = {'tmax': 'Daily Maximum Temperature',
             'tmin': 'Daily Minimum Temperature',
             'pr': 'Daily Precipitation',
             'prec': 'Daily Precipitation'}

#==============================================================================
"""Creating a Plot class to minimize data-passing"""
#==============================================================================

class Plot:
    """
        Stores appropriate data. Currently no methods.

        Slots:
            plot_path : the folder in which to save the plots (str)
            lat : latitude (float)
            lon : longitude (float)
            predictand : the name of the variable to predict,
                must match the name of a col in the obs xarray object (str)
            obs : the observed values (xarray object)
            models : a dictionary with the model names as keys and
                xarray objects containing predictions as the values.
                These xarray objects must contain a "preds" column (dictionary)
            startYr : first year of data (int)
            endYr : last year of data (int)
    """

    __slots__ = ['plot_path', 'lat', 'lon', 'predictand', 'obs', 'models', 'startYr','endYr', 'k']

    def __init__(self, save_path, lat, lon, predictand, obs, models,
                 startDate, endDate, k):
        # create a folder to save the plots
        folder = "plots"
        try:
            os.mkdir(os.path.join(save_path, folder))
        except:
            pass #assume folder exists
        self.plot_path =  os.path.join(save_path, folder)
        # setting the rest of the slots
        self.lat, self.lon = lat, lon
        self.predictand = predictand
        self.obs = obs
        self.models = models #a dict of xarray objs; must have preds as a var
        self.startYr, self.endYr = int(startDate[:4]), int(endDate[:4])
        #to match obs
        self.startYr = max(self.startYr, 1980)
        self.k = k


#===============================================================================
"""
Mehtods to create seasonal line plots.
"""
#===============================================================================

def annualSeasonPlot(plotData, startDate, endDate, title):
    """
        Plots seasonal avgs compared to an observed moving average
        Input:
            plotData, a Plot object
            startDate of season as a string; ex. '12-01'
            endDate of season as a string; ex. '02-28'
            title as a string; ex. 'Season plot: winter'
        Output: None
    """
    # if winter, the dates wrap around years
    winter = startDate[0:2] > endDate[0:2]

    #set date string to slice data
    startDate, endDate = '-' + startDate, '-' + endDate
    startYr = plotData.startYr
    endYr = plotData.endYr

    #create time series for plotting
    obsSeasonAvg = [plotData.obs.sel(time = slice(str(y)+startDate, str(y+winter) + endDate))[plotData.predictand].mean(dim = 'time') for y in range(startYr, endYr+1) ]
    modelSeasonAvg = dict()
    for model in plotData.models.keys():
        modelSeasonAvg[model] = [plotData.models[model].sel(time = slice(str(y)+startDate, str(y+winter)+endDate)).preds.mean(dim = 'time') for y in range(startYr, endYr+1) ]

    movingAvg = [sum([obsSeasonAvg[t+i] for i in range(5)])/5 for t in range(len(obsSeasonAvg)-4)]

    #plot lines
    plt.plot(range(startYr, endYr + 1), obsSeasonAvg, label = 'obs')
    for model in plotData.models.keys():
        plt.plot(range(startYr, endYr + 1), modelSeasonAvg[model], label=model)
    plt.plot(range(startYr+2, endYr-1), movingAvg, '-k', label = "5-yr moving average")

    #label and save figure
    plt.suptitle(title)
    plt.title(titleDict[plotData.predictand])
    if plotData.predictand in ['tmin', 'tmax']:
        plt.ylabel('Temperature (Celcius)')
    if plotData.predictand == 'prec':
        plt.ylabel('Precipitation (mm)')
    plt.xlabel('Year')
    plt.legend()
    plt.savefig(os.path.join(f"{plotData.plot_path}/seasonalPlots", f"{title.replace(' ','')}.pdf"))
    plt.clf()


def plot_all_seasons(plotData):
    """
        Plots 4 plots of seasonal avgs compared to an observed moving average
        Input:
            plotData, a Plot object with necessary data
        Output: None
    """
    plotData.obs['time'] = plotData.obs['timecopy']
    annualSeasonPlot(plotData, '12-01', '02-28', "Seasonal Plot Dec-Jan-Feb")
    annualSeasonPlot(plotData, '03-01', '05-30', "Seasonal Plot Mar-Apr-May")
    annualSeasonPlot(plotData, '06-01', '08-31', "Seasonal Plot Jun-Jul-Aug")
    annualSeasonPlot(plotData, '09-01', '11-30', "Seasonal Plot Sep-Oct-Nov")




#==============================================================================
"""Time series plots."""
#==============================================================================

def plot_monthly_avgs(plotData):
    """
        Saving a plot of monthly averages for predictand.
        Input: plotData object
        Output: None
    """
    #set years
    startYr = plotData.startYr
    endYr = plotData.endYr

    #condition on month and create time series
    plotData.obs['time'] = plotData.obs['month']
    modelAvgs = dict()
    for model in plotData.models.keys():
        plotData.models[model]['time'] = plotData.models[model].time.dt.month
        modelAvgs[model] = [float(plotData.models[model].sel(time = m).mean(dim = 'time')['preds']) for m in range(1,13)]
    obsAvgs = [float(plotData.obs.sel(time = m).mean(dim = 'time')[plotData.predictand]) for m in range(1,13)]

    #plot lines
    plt.plot(monthsAbrev, obsAvgs, label = 'obs')
    for model in plotData.models.keys():
        plt.plot(monthsAbrev, modelAvgs[model], label=model)

    #label and save figure
    plt.suptitle("Monthly Means for Observed and Modeled Data")
    plt.title(titleDict[plotData.predictand])
    if plotData.predictand in ['tmin', 'tmax']:
        plt.ylabel('Temperature (Celcius)')
    if plotData.predictand == 'prec':
        plt.ylabel('Precipitation (mm)')
    plt.xlabel('Month')
    plt.legend()
    plt.savefig(os.path.join(f"{plotData.plot_path}/timeSeriesPlots", 'monthly_means.pdf'))
    plt.clf()

def plot_daily_avgs(plotData):
    """
        Saving a plot of daily averages for predictand.
        Input: plotData object
        Output: None
    """
    #condition on month and create time series
    modelAvgs = dict()
    for model in plotData.models.keys():
        data = plotData.models[model]
        data['time'] = data.timecopy
        modelAvgs[model] = [float(data.sel(time = (data.time.dt.month == m) & (data.time.dt.day == d)).mean(dim = 'time')['preds']) for m,d in zip(data.time.dt.month[366:(366+365)], data.time.dt.day[366:(366+365)])]
    obs = plotData.obs
    obs['time'] = obs.timecopy
    obsAvgs = [float(obs.sel(time = (obs.time.dt.month == m) & (obs.time.dt.day == d)).mean(dim = 'time')[plotData.predictand]) for m,d in zip(obs.time.dt.month[366:(366+365)], obs.time.dt.day[366:(366+365)])]

    #plot lines
    days = [f"{m}-{d}" for m,d in zip(obs.time.dt.month[366:(366+365)], obs.time.dt.day[366:(366+365)])]
    plt.plot(days, obsAvgs, label = 'obs')
    for model in plotData.models.keys():
        plt.plot(days, modelAvgs[model], label=model)

    #label and save figure
    plt.suptitle("Daily Means for Observed and Modeled Data")
    plt.title(titleDict[plotData.predictand])
    if plotData.predictand in ['tmin', 'tmax']:
        plt.ylabel('Temperature (Celcius)')
    if plotData.predictand == 'prec':
        plt.ylabel('Precipitation (mm)')
    plt.xlabel('Day')
    x = np.arange(1)
    plt.xticks(x, "")
    plt.legend()
    plt.savefig(os.path.join(f"{plotData.plot_path}/timeSeriesPlots", 'daily_means.pdf'))
    plt.clf()


def plot_annual_avgs(plotData):
    """
        Saving a plot of annual averages for predictand.
        Input: plotData object
        Output: None
    """

    # condition on year
    startYr = plotData.startYr
    endYr = plotData.endYr
    plotData.obs['time'] = plotData.obs.timecopy.dt.year
    modelAvgs = dict()
    for model in plotData.models.keys():
        plotData.models[model]['time'] = plotData.models[model].timecopy.dt.year

        #create time series of annual averages
        modelAvgs[model] = [float(plotData.models[model].sel(time = yr).mean(dim = 'time')['preds']) for yr in range(startYr, endYr+1)]
    obsAvgs = [float(plotData.obs.sel(time = yr).mean(dim = 'time')[plotData.predictand]) for yr in range(startYr, endYr+1)]

    #plot data
    plt.plot(range(startYr, endYr+1), obsAvgs, label = 'obs')
    for model in plotData.models.keys():
        plt.plot(range(startYr, endYr+1), modelAvgs[model], label=model)

    #label plot
    plt.suptitle("Annual Means for Observed and Modeled Data")
    plt.title(titleDict[plotData.predictand])
    if plotData.predictand in ['tmin', 'tmax']:
        plt.ylabel('Temperature (Celcius)')
    if plotData.predictand == 'prec':
        plt.ylabel('Precipitation (mm)')
    plt.xlabel('Year')
    plt.legend()

    #save
    plt.savefig(os.path.join(f"{plotData.plot_path}/timeSeriesPlots", 'annual_means.pdf'))
    plt.clf()

def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    Source: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    return ax

def plot_annual_avgs_bar(plotData):
    """
        Saving a plot of annual averages for predictand.
        Input: plotData object
        Output: None
    """
    # get year data
    startYr = plotData.startYr
    endYr = plotData.endYr
    plotData.obs['time'] = plotData.obs.timecopy.dt.year
    modelAvgs = dict()
    #take just first model
    model = [key for key in plotData.models.keys()][0]
    plotData.models[model]['time'] = plotData.models[model].timecopy.dt.year

    #calculate annual means
    modelAvgs = [float(plotData.models[model].sel(time = yr).mean(dim = 'time')['preds']) for yr in range(startYr, endYr+1)]
    obsAvgs = [float(plotData.obs.sel(time = yr).mean(dim = 'time')[plotData.predictand]) for yr in range(startYr, endYr+1)]

    #labels are years with data
    labels = range(startYr, endYr+1)
    x = np.arange(len(labels))
    width = 5 / len(labels) #the width of the bars

    # set barplot objects
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, obsAvgs, width, label = "obs")
    rects2 = ax.bar(x + width/2, modelAvgs, width, label = "model")

    # create appropriate labels
    ax.set_ylabel(f"Annual Mean {plotData.predictand}")
    ax.set_title("Annual Means")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # label data values for heights of bars
    #ax = autolabel(rects1, ax)
    #ax = autolabel(rects2, ax)
    fig.tight_layout()

    #save plot
    plt.savefig(os.path.join(f"{plotData.plot_path}/timeSeriesPlots", 'annual_means_bar.pdf'))
    plt.clf()

def plot_cond_days(plotData, title, comp = "greater", thresh = 35):
    """
        Saving a plot of days that satisify cond.
        Ex. number of days over 35 degrees C
        Input: plotData, Plot obj
               title - a str plot title and plot file name
               comp - "greater" or "less" for comparison to the threshold
               thresh - cutoff for days to count (inclusive)
        Output: None
    """
    #find number of years
    yearCount = plotData.endYr - plotData.startYr + 1

    # condition data on month
    plotData.obs['time'] = plotData.obs.month
    for key in plotData.models.keys():
        plotData.models[key]['time'] = plotData.models[key].month

    # count number of days filling criteria by month to create time series
    if comp.lower() == "greater": #greater than or equal to thresh
        obsDaysCount = [sum(plotData.obs.sel(time=m)[plotData.predictand].values >= thresh)/yearCount for m in range(1,13)]
        modelDaysCount = dict()
        for model in plotData.models.keys():
            modelDaysCount[model] = [sum(plotData.models[model].sel(time=m).preds.values >= thresh)/yearCount for m in range(1,13)]
    else: #less than or equal to thresh
        obsDaysCount = [sum(plotData.obs.sel(time=m)[plotData.predictand].values <= thresh)/yearCount for m in range(1,13)]
        modelDaysCount = dict()
        for model in plotData.models.keys():
            modelDaysCount[model] = [sum(plotData.models[model].sel(time=m).preds.values <= thresh)/yearCount for m in range(1,13)]

    #plot data as a line plot
    plt.plot(monthsAbrev, obsDaysCount, label = 'obs')
    for model in plotData.models.keys():
        plt.plot(monthsAbrev, modelDaysCount[model], label=model)

    # label plot
    plt.suptitle(title)
    plt.title(titleDict[plotData.predictand])
    plt.ylabel('Number of Days')
    plt.xlabel('Month')
    plt.legend()

    #save figure
    plt.savefig(os.path.join(f"{plotData.plot_path}/timeSeriesPlots", f"{title.replace(' ','')}.pdf"))
    plt.clf()

def plot_cond_days_by_year(plotData, title, comp = "greater", thresh = 35):
    """
        Saving a plot of days that satisify cond.
        Ex. number of days over 35 degrees C
        Input: plotData, Plot obj
               title - a str plot title and plot file name
               comp - "greater" or "less" for comparison to the threshold
               thresh - cutoff for days to count (inclusive)
        Output: None
    """

    # condition data on year
    plotData.obs['time'] = plotData.obs.timecopy.dt.year
    for key in plotData.models.keys():
        plotData.models[key]['time'] = plotData.models[key].timecopy.dt.year

    # count number of days filling criteria by month to create time series
    if comp.lower() == "greater": #greater than or equal to thresh
        obsDaysCount = [sum(plotData.obs.sel(time=y)[plotData.predictand].values >= thresh) for y in range(plotData.startYr, plotData.endYr + 1)]
        modelDaysCount = dict()
        for model in plotData.models.keys():
            modelDaysCount[model] = [sum(plotData.models[model].sel(time=m).preds.values >= thresh) for m in range(plotData.startYr, plotData.endYr + 1)]
    else: #less than or equal to thresh
        obsDaysCount = [sum(plotData.obs.sel(time=m)[plotData.predictand].values <= thresh) for m in range(plotData.startYr, plotData.endYr + 1)]
        modelDaysCount = dict()
        for model in plotData.models.keys():
            modelDaysCount[model] = [sum(plotData.models[model].sel(time=m).preds.values <= thresh) for m in range(plotData.startYr, plotData.endYr + 1)]

    #plot data as a line plot
    plt.plot(range(plotData.startYr, plotData.endYr + 1), obsDaysCount, label = 'obs')
    for model in plotData.models.keys():
        plt.plot(range(plotData.startYr, plotData.endYr + 1), modelDaysCount[model], label=model)

    # label plot
    plt.suptitle(title)
    plt.title(titleDict[plotData.predictand])
    plt.ylabel('Number of Days')
    plt.xlabel('Year')
    plt.legend()

    #save figure
    plt.savefig(os.path.join(f"{plotData.plot_path}/timeSeriesPlots", f"{title.replace(' ','')}.pdf"))
    plt.clf()

def plot_hot_days(plotData, future = False):
    """
       Saving a plot of days over 35 degrees C
       Input: plotData, a Plot object with necessary data
       Output: None
    """
    plot_cond_days(plotData, title="Number of Days at Least 35 Degrees Celcius by Month", comp="greater", thresh=35)
    if not future:
        plot_cond_days_by_year(plotData, title="Number of Days at Least 35 Degrees Celcius by Year", comp="greater", thresh=35)

def plot_cold_days(plotData, future = False):
    """ todo """
    plot_cond_days(plotData, title="Number of Freezing Days by Month", comp="less", thresh = 0)
    if not future:
        plot_cond_days_by_year(plotData, title="Number of Freezing Days by Year", comp="less", thresh = 0)

#==============================================================================
"""Plotting distributions:
        -histograms,
        -boxplots,
        -violin plots"""
#==============================================================================

def plot_dist(plotData, data, title):
    """
        Generates and saves a plot of the distribution of given data
        Input:
            plotData, a Plot object with necessary data
            data, a str of what will be plotted, ie. "plotData.obs[plotData.predictand]"
            title, a string to appear on plot
        Output:
            none
    """
    # plot histogram; weight by proportion of points
    #from matplotlib.ticker import FuncFormatter, PercentFormatter
    plt.hist(eval(data),
             alpha = 0.5, #make translucent
             bins = 25,
             weights = np.ones(len(eval(data).values)) / (len(eval(data).values)))

    #label plot
    # plt.suptitle(title)
    # plt.title(titleDict[plotData.predictand])
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.ylabel("Percent")
    # #plt.xlim(-20,50)
    # plt.xlabel(plotData.predictand)
    #
    # #save figure
    # plt.savefig(os.path.join(f"{plotData.plot_path}/distributionPlots", f"{title.replace(' ','')}.pdf"))
    # plt.clf()

def plot_dists(plotData):
    """
        Plots obs and model histograms
    """
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    #label plot
    plt.suptitle(f"Distributions of Observations and Predictions")
    plt.title(titleDict[plotData.predictand])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel("Percent")
    #plt.xlim(-20,50)
    plt.xlabel(plotData.predictand)

    #plot observed distribution
    plot_dist(plotData, "plotData.obs[plotData.predictand]")
    # plot any model distributions
    for model in plotData.models.keys():
        plot_dist(plotData, f"plotData.models['{model}'].preds")

    #save figure
    plt.savefig(os.path.join(f"{plotData.plot_path}/distributionPlots", "histograms.pdf"))
    plt.clf()


def boxplot(plotData):
    """ creates a boxplot of obs and any model data"""
    labels = ['obs'] + list(plotData.models.keys())
    data = [plotData.obs[plotData.predictand]] + [m.preds for m in plotData.models.values()]
    plt.boxplot(data, whis = 0.75, labels = labels)
    plt.suptitle("Boxplots for Observed and Modeled Data")
    plt.title(titleDict[plotData.predictand])

    #y axis label
    if plotData.predictand in ['tmin', 'tmax']:
        plt.ylabel('Degrees Celcius')
    else:
        plt.ylabel('mm')

    plt.savefig(os.path.join(f"{plotData.plot_path}/distributionPlots", "boxplots.pdf"))

def violin(plotData):
    """ creates a violin plot of obs and any model data"""
    labels = ['obs'] + list(plotData.models.keys())
    data = [plotData.obs[plotData.predictand]] + [m.preds for m in plotData.models.values()]
    fig, ax = plt.subplots()

    #set up x-axis
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

    #make plot
    plt.violinplot(data)
    plt.suptitle("Violin Plots for Observed and Modeled Data")
    plt.title(titleDict[plotData.predictand])

    #y axis label
    if plotData.predictand in ['tmin', 'tmax']:
        plt.ylabel('Degrees Celcius')
    else:
        plt.ylabel('mm')

    plt.savefig(os.path.join(f"{plotData.plot_path}/distributionPlots", "violinplots.pdf"))



#===============================================================================
"""
Saving summary statistics.
"""
#===============================================================================

def RMSE(preds, obs):
    """Input: numpy matrix of predictions, numpy matrix of true values (for both, each row is a day)
       Output: Root mean squared error, as a float"""
    return np.sqrt(np.square(preds.flatten() - obs.flatten()).mean())

def AIC(preds, obs, k):
    """Input: numpy matrix of predictions,
              numpy matrix of true values (for both, each row is a day)
              k (integer), the number of predictors
       Output: Akaike's Information Criterion, as a float"""
    n = preds.shape[0] #length of time series (number of points)
    SSE = np.square(preds.flatten() - obs.flatten()).sum() #Sum of squared errors
    return (2*k) + (n * np.log(SSE/n))

def Rsq(preds, obs):
    """to do"""
    #SSE = np.square(preds.flatten() - obs.flatten()).sum() #Sum of squared errors
    ybar = float(obs.flatten().mean())
    SSR = np.square(preds.flatten() - ybar).sum()
    SST = np.square(obs.flatten() - ybar).sum() #Sum of squares total
    #from sklearn.metrics import r2_score
    #return r2_score(obs, preds)
    return (SSR / SST)

def adjustedRsq(preds, obs, k):
    """adjusted R squared"""
    n = preds.shape[0] #length of time series (number of points)
    return 1 - ( (1-Rsq(preds, obs)) * (n-1) ) / (n-k-1)

def save_stats(plotData):
    """
        Saving a txt file with data on the modeled predictions
        Input: plotData
        Output: None
        # TODO: r-squared; standard error
    """
    #open file
    f = open(os.path.join(plotData.plot_path, "summary.txt"), "w")

    f.write("Summary Statistics\n")
    f.write(f"Lat = {plotData.lat}, Lon = {plotData.lon}\n\n")

    #write stats for observed data
    f.write("Observations:\n")
    f.write(f"Mean: {float(np.mean(plotData.obs[plotData.predictand].values))}\n")
    f.write(f"Variance: {float(np.var(plotData.obs[plotData.predictand].values))}\n\n")

    #write stats for any modeled data
    for key in plotData.models.keys():
        f.write(f"Modeled Data ({key}):\n")
        f.write(f"Mean: {float(np.mean(plotData.models[key].preds.values))}\n")
        f.write(f"Variance: {float(np.var(plotData.models[key].preds.values))}\n")
        f.write(f"RMSE: {RMSE(plotData.models[key].preds.values, plotData.obs[plotData.predictand].values)}\n")
        # todo fix k for multiple models
        f.write(f"AIC: {AIC(plotData.models[key].preds.values, plotData.obs[plotData.predictand].values, plotData.k)}\n")
        f.write(f"R-squared: {Rsq(plotData.models[key].preds.values, plotData.obs[plotData.predictand].values)}\n")
        f.write(f"Adjusted R-squared: {adjustedRsq(plotData.models[key].preds.values, plotData.obs[plotData.predictand].values, plotData.k)}\n\n")
    f.close()


#===============================================================================
"""Creating all possible plots"""
#===============================================================================


def plot_all(plotData):
    """ Creates all available plots organized in three folders"""

    #create folders to save plots
    for folder in ['seasonalPlots', 'distributionPlots', 'timeSeriesPlots']:
        try:
            os.mkdir(os.path.join(plotData.plot_path, folder))
        except: pass

    #seasonal plots
    plot_all_seasons(plotData)

    #time series plots
    plot_monthly_avgs(plotData)
    if plotData.predictand == 'tmax':
        plot_hot_days(plotData)
    elif plotData.predictand == 'tmin':
        plot_cold_days(plotData)
    plot_annual_avgs(plotData)
    #plot_annual_avgs_bar(plotData)
    plot_daily_avgs(plotData)

    #summary statistics
    save_stats(plotData)

    #plot distributions
    plot_dists(plotData)
    boxplot(plotData)
    violin(plotData)


def plot_all_future(plotData):
    """ Creates all possible plots for future data organized in three folders"""

    #create folders to save plots
    for folder in ['seasonalPlots', 'distributionPlots', 'timeSeriesPlots']:
        try:
            os.mkdir(os.path.join(plotData.plot_path, folder))
        except: pass

    #time series plots
    plot_monthly_avgs(plotData)
    if plotData.predictand == 'tmax':
        plot_hot_days(plotData, True)
    elif plotData.predictand == 'tmin':
        plot_cold_days(plotData, True)
    plot_daily_avgs(plotData)

    #plot distributions
    plot_dists(plotData)
    boxplot(plotData)
    violin(plotData)
