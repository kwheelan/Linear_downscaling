
import xarray as xr
import numpy as np

f = open('out.txt', 'w')
preds = xr.open_dataset('/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc').tmax
#'/glade/scratch/kwheelan/downscaling_data/preds/finalPreds_tmax_38.125_-101.875.nc')
preds['time'] = preds.time.dt.month

for month in range(0, 13):
    if month == 0:
#        line = ', '.join(preds.keys())
        f.write(f"month, mean, var")
    else:
        f.write(f"\n{month}, {np.mean(preds.sel(time=month).values)}, {np.var(preds.sel(time=month).values)}")
        """for key in preds.keys():
            try:
                mean = np.var(preds.sel(time=month)[key].values)
                f.write(f"{mean}, ")   
            except:
                pass

#get annual stats
f.write("\nAnnual, ")
for key in preds.keys():
    try:
        mean = np.var(preds[key].values)
        f.write(f"{mean}, ")
    except: pass
"""
f.close()
