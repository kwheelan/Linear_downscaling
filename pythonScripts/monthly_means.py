
import xarray as xr
import numpy as np

f = open('out.txt', 'w')
preds = xr.open_dataset('/glade/scratch/kwheelan/downscaling_data/preds/finalPreds_tmax_38.125_-101.875.nc')
preds['time'] = preds.month

for month in range(0, 13):
    if month == 0:
        line = ', '.join(preds.keys())
        f.write(f"month, {line}")
    else:
        f.write(f"\n{month}, ")
        for key in preds.keys():
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

f.close()
