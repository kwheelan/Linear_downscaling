
#import modules
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mlab
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

#open data
sample_data = xr.open_dataset("/glade/work/rmccrary/SnowCOMP/data/sample/sample_data.nc", decode_times=False)

time_range = [9,10,11] + list(range(9)) #year starts with october

#get two data series
future = [sample_data.SCE_Future.sel(time = i).data for i in time_range]
historical = [sample_data.SCE_Historical.sel(time = i).data for i in time_range]

def plots(data, title, boxplot=False):
    """A fn for plotting violinplots and boxplots"""
    plt.clf() #clear plot window
    fig, ax = plt.subplots(figsize=(8, 4)) #dimensions of the plotting window
    if boxplot:
        plt.boxplot(data) #add boxplots
    else:
        plt.violinplot(data) #add violins

    #add markers for the means
    means = [np.mean(data[i]) for i in range(12)]
    plt.plot(range(1,13), means, marker = "o")
    
    #labels
    plt.title("Violin Plots for Sample Data")
    plt.xlabel("Month")
    plt.ylabel("SCE")
    #these two lines below change the x-axis labels to match the actual months (starting with oct)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels([i+1 for i in time_range])
#   ax.set_xticklabels(["Oct", "Nov", .....])
    plt.savefig(f"{title}.png")

plots(future, "future_violins")
plots(historical, "hist_violins")
plots(future, "future_boxplots", True)
plots(historical, "hist_boxplots", True)

def lineplot(data, title):
    plt.clf() #clear plot
    fig, ax = plt.subplots(figsize=(8, 4)) #set up plot window
    for i in range(18):
        #draw each line
        plt.plot(range(12), [data[j][i] for j in range(12)]) #range(12) goes from month 0 to 11 (oct to sept); range(1,12) goes nov thru sept
    #labels
    plt.title("Lineplot")
    plt.xlabel("Month")
    #these two lines below change the x-axis labels to match the actual months (starting with oct)
    ax.set_xticks(range(12)) 
    ax.set_xticklabels([i+1 for i in time_range])
    plt.ylabel("SCE")
    #save plot
    plt.savefig(f"{title}.png")

lineplot(future, "future_line")
lineplot(historical, "hist_line")



