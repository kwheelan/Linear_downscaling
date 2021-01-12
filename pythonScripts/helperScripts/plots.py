import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mlab
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from random import random, randint

obs = xr.open_dataset('/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc').sel(lat = 38.125, lon=-101.875, method = 'nearest').sel(time = slice('1980-01-01','2005-12-31'))
obs = obs.tmax.values
SDSMdata = [ [float(line.strip()) for line in open(f"/glade/work/kwheelan/Linear_downscaling/SDSM_p1/tmax/preds/tmax{n}.OUT")] for n in range(1,6)]
SDSMdata += [obs]

labels = list(range(1,6)) + ['obs']

pythonData = [xr.open_dataset("")]
#[ xr.open_dataset(f"/glade/scratch/kwheelan/downscaling_data/runs/run{n}/tmax_lat38.125_lon-101.875/preds/finalPreds_tmax_38.125_-101.875.nc").preds.values for n in range(1,6) ]
pythonData += [obs]

def histogram(data, title, fp):
#     plt.clf()
     plt.hist(data, bins=50, density=1)
     plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
     plt.title(title)
     plt.xlim(-15,15)
     plt.ylim(0, .1)
     plt.savefig(fp)

fp = '/glade/work/kwheelan/Linear_downscaling/SDSM_p1/tmax/preds/tmax_p1_output_no_var_inflation.OUT'
preds = [float(line.strip()) for line in open(fp)]

pythonDiff = []
SDSM_diff = []
for i in range(5):
     SDSM_diff += [y - p for y,p in zip(SDSMdata[i], preds)]
     pythonDiff += [y - p for y,p in zip(pythonData[i], preds)]

n = len(preds)
stderrors = np.std(obs - preds)
SE = stderrors * np.sqrt((n-1)/(n-2))

def vi(SE, c=12):
    Ri = sum([random() for i in range(c)])
    return SE * (Ri - (c/2))

fig, ax = plt.subplots(figsize=(8, 4))
bins = 1000
ax.hist(np.array(SDSM_diff), density=1, bins=bins, histtype='step', cumulative = "True", label = "SDSM v_i CDF")
ax.hist(np.array(pythonDiff), density=1, bins=bins, histtype='step', cumulative = "True", label = "Python v_i CDF")

from scipy.stats import norm, gaussian_kde
y= norm.pdf(np.linspace(-15, 15, bins+1),  0, SE).cumsum()
y  /= y[-1]
#ax.plot(np.linspace(-15, 15, bins+1), y, 'k--', linewidth=1.5, label='Theoretical Normal')
plt.legend(loc = 'upper left')
plt.title("Cumulative Density Plots")
plt.savefig('CDF.png')
plt.clf()

def dists(data1, data2, lo, hi, bins, title):
     plt.clf()
     fig, ax = plt.subplots(figsize=(8, 4))
     n, x, _ = ax.hist(data1, density=1, bins=bins, histtype='step')
     n2, x2, _ = ax.hist(np.array(data2), density=1, bins=bins, histtype='step')
     y= norm.pdf(np.linspace(lo, hi, bins), np.mean(data1), np.std(data1))
     bin_centers = 0.5*(x[1:]+x[:-1])
     bin_centers2 = 0.5*(x2[1:]+x2[:-1])
     plt.clf()
     plt.plot(bin_centers, n, label = 'SDSM')
     plt.plot(bin_centers2, n2, label = 'Python generated')
     plt.plot(np.linspace(lo, hi, bins), y, label = 'Theoretical normal', linestyle = '--', color = 'k')
     plt.legend(loc = 'upper left')
     plt.title("Distributions of Variance Inflation Terms (n = 47,485)")
     plt.xlim(lo, hi)
     plt.savefig(title)


#histogram(pythonDiff, "Distributions", 'hists.png')
#histogram(SDSM_diff, "Distributions", 'hists.png')

from random import randint
new_preds = np.array(preds) + np.array([SDSM_diff[randint(0,len(SDSM_diff)-1)] for i in range(len(preds))])
#dists(np.array(SDSMdata[0]), new_preds, -20, 50, bins=100, title= "Sampling_error")
#dists(SDSM_diff, [vi(SE) for i in range(47485)], -15, 15, 60, 'var_infl_dists')

plt.clf()
plt.boxplot(SDSMdata, whis=0.75, labels=labels)
plt.title('Distributions of 5 SDSM runs -- variance inflation = 12')
plt.ylim(-25, 50)
#plt.savefig('SDSM_boxplots')

plt.clf()
plt.boxplot(pythonData, whis=0.75, labels=labels)
plt.title("Distributions of 5 Python runs -- variance inflation = 12")
plt.ylim(-25,50)
#plt.savefig('python_boxplots')

plt.clf()

mslp = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/mslp_ERAI_NAmerica_surf_19790101-20181231_dayavg_mpigrid.nc')
mslp = mslp.sel(lat = 38.125, lon = -101.875, method='nearest').sel(time = slice('1980-01-01', '2005-12-31'))
fig, ax = plt.subplots(figsize=(8, 4))

labels = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mslp['time'] = mslp.time.dt.month
data = [mslp.mslp.sel(time = m).values for m in range(1,13)]

#set up x-axis
ax.get_xaxis().set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, len(labels) + 1))
ax.set_xticklabels(labels)
ax.set_xlim(0.25, len(labels) + 0.75)

#make plot
plt.violinplot(data)
plt.title("Violin Plots for Monthly Mean Sea Level Pressure")
means = [np.mean(data[i]) for i in range(12)]
plt.plot(list(range(1,13)), means, marker = "o")
plt.xlabel("Month")
plt.ylabel("MSLP (Pa)")
#plt.savefig("mslp_violinplots.png")
