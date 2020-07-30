
#==============================================================================
"""A script to process and plot SDSM output for comparison."""
#==============================================================================

#import packages
import xarray as xr
import pandas as import pd

lat = 38.125
lon = -101.875
obsPath = '/glade/p/cisl/risc/narccap/obs/gridMET/common/DCA/tmax.gridMET.NAM-22i.SGP.nc'
save_path = '/glade/scratch/$USER/downscaling_data/SDSM_p1'
pred_file = 'tmax_p1_output.OUT'
predictand = 'tmax'

sdsm_preds = pd.read_csv(pred_file)

obs = xr.open_dataset(obsPath).sel(time = slice(dateStart, dateEnd))
Y_all = obs.sel(lat = lat, lon = lon, method = 'nearest').sel(time = slice(dateStart, dateEnd))

preds  = xr.DataArray(Y_all.time)
preds['month'] = Y_all.time.month
preds['preds'] = sdsm_preds

final_predictions = preds

plot_all_seasons(Y_all, final_predictions, save_path, lat, lon, predictand)
plot_monthly_avgs(Y_all, final_predictions, save_path, lat, lon, predictand)
plot_hot_days(Y_all, final_predictions, save_path, lat, lon)
save_stats(Y_all, final_predictions, lat, lon, save_path, predictand)
plot_dist(Y_all[predictand], 'Observed Distribution', save_path, lat, lon, predictand)
plot_dist(final_predictions.preds, 'SDSM Modeled Distribution', save_path, lat, lon, predictand)
