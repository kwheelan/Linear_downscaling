import xarray as xr
import mathplotlib.pyplot as plt
import matplotlib as mlab
import numpy as np

obs = xr.open_dataset('/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc').sel(lat = 38.125, lon=-101.875, method = 'nearest').sel(time = slice('1980-01-01','2005-12-31'))
SDSMdata = [ [float(line.strip()) for line in open(f"tmax{n}.OUT")] for n in range(1,6)]
SDSMdata += [obs]

labels = list(range(1,6)) + ['obs']

pythonData = [ xr.open_dataset(f"/glade/scratch/kwheelan/downscaling_data/runs/run{n}/tmax_lat38.125_lon-101.875//preds/finalPreds_tmax_38.125_-101.875.nc").preds.values for n in range(1,6) ]
pythonData += [obs]

def histogram(data, title, fp):
#     plt.clf()
     plt.hist(data, bins=50, weights = np.ones(len(data))/len(data))
     plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
     plt.title(title)
     plt.xlim(-15,15)
     plt.ylim(0, .1)
     plt.savefig(fp)

fp = '../SDSM_p1/tmax/preds/tmax_p1_output_no_var_inflation.OUT'
preds = [float(line.strip()) for line in open(fp)]

pythonDiff = []
SDSM_diff = []
for i in range(5):
     SDSM_diff += [y - p for y,p in zip(SDSM_data[i], preds)]
     pythonDiff += [y - p for y,p in zip(pythonData[i], preds)]

n = len(preds)
stderrors = np.std(obs.tmax - preds)
SE = stderrors * np.sqrt((n-1)/(n-2))

def vi(SE, c=12):
    Ri = sum([random() for i in range(c)])
    return SE * (Ri - (c/2))

fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(np.array(SDSM_diff)/sum(SDSM_diff), 50, histtype='step', cumulative = "True", label = "SDSM")
n, bins, patches = ax.hist(np.array(SDSM_diff)/SDSM_diff.cumsum()[-1], 50, histtype='step', cumulative = "True", label = "Python")
y= np.random.normal(0, SE, 51).cumsum()
y  /= y[-1]
ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
plt.legend()
plt.savefig('test')
plt.clf()

histogram(pythonDiff, "Distributions", )
