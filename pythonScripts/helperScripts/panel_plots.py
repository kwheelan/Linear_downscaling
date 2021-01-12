import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

mslp = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/mslp_ERAI_NAmerica_surf_19790101-20181231_dayavg_mpigrid.nc').sel(time = slice('1980-01-01', '2005-12-31'))
vas = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/vas_ERAI_NAmerica_surf_19790101-20181231_dayavg_mpigrid.nc').sel(time = slice('1980-01-01', '2005-12-31'))
Z_p500 = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/Z_ERAI_NAmerica_p500_19790101-20181231_dayavg_mpigrid.nc').sel(time = slice('1980-01-01', '2005-12-31'))
Z_p700 = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/Z_ERAI_NAmerica_p700_19790101-20181231_dayavg_mpigrid.nc').sel(time = slice('1980-01-01', '2005-12-31'))
RH_p500 = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/RH_ERAI_NAmerica_p500_19790101-20181231_dayavg_mpigrid.nc').sel(time = slice('1980-01-01', '2005-12-31'))
RH_p850 = xr.open_dataset('/glade/p/cisl/risc/rmccrary/DOE_ESD/LargeScale_DCA/ERA-I/mpigrid/RH_ERAI_NAmerica_p850_19790101-20181231_dayavg_mpigrid.nc').sel(time = slice('1980-01-01', '2005-12-31'))

labels = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def select_coord(ds, lat, lon):
    return ds.sel(lat = lat, lon=lon, method = "nearest")

def get_month_data(data, pred):
    avgs = [data[pred].sel(time = m).values for m in range(1,13)]
    return avgs

def normalize(ds, pred, lat, lon):
    var = select_coord(ds, lat, lon)[pred]
    zscores = (var - np.mean(var.values)) / np.std(var.values)
    var[pred] = (('time'), zscores)
    return var

def make_plot(ds, pred, title, norm):
    ds['time'] = ds.time.dt.month

    plt.clf()
    fig, axs = plt.subplots(4, 5, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})

    for ax in axs.flat:
        ax.label_outer()

    for row in range(len(axs)):
        for col in range(len(axs[0])):
            lat = 38.125 - row*2
            lon = -101.875 + 2*col
            if norm:
                data = normalize(ds, pred, lat, lon)
            data = get_month_data(data, pred)
            axs[row][col].violinplot(data)
            means = [np.mean(data[i]) for i in range(12)]
            axs[row][col].plot(list(range(1,13)), means)#, marker = "o")
            axs[row][col].hlines(y=0, xmin=1, xmax=12, colors = 'k', linestyles = 'dotted')
    var_name = 'Normalized ' * norm + title
    fig.suptitle(f"{var_name} by Month for 20 Lat-Lon Points")
    fig.text(0.5, 0.04, 'Month', ha='center')
    #fig.text(0.04, 0.5, 'MSLP (normalized)', va='center', rotation='vertical')
    plt.savefig(f"{title}_norm.png")

for ds,pred,title in zip([mslp, vas, Z_p500, Z_p700, RH_p500, RH_p850], ['mslp', 'vas', 'Z', 'Z', 'RH', 'RH'], ['MSLP', 'VAS', 'Z_p500', 'Z_p700', 'RH_p500', 'RH_p850']):
#    make_plot(ds, pred, title, False)
    make_plot(ds, pred, title, True)
