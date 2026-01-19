import sys

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import xarray as xr
from loess.loess_1d import loess_1d
from geopy import distance

lowess = sm.nonparametric.lowess


def fill_missing_nearest(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr


def count_np(arr):
    np.count_nonzero(~np.isnan(arr))


sat = "S3A"
track_number = "195"
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

f_track = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number}.nc"
ds = xr.open_dataset(f_track, decode_times=False)
print(f_track)
print(ds)
lons_track = ds.lon.values
lats_track = ds.lat.values
lon_s = lons_track[0]
lon_e = lons_track[-1]
lat_s = lats_track[0]
lat_e = lats_track[-1]
total_dist = distance.distance((lat_s, lon_s), (lat_e, lon_e)).km
frac = total_dist / 40.0  # km
print(total_dist, frac)
x = ds.points_numbers.values
gshhs = ds.dist_to_coast_gshhs.values
sla_track = ds.sla
sla_track1 = sla_track.isel(cycles_numbers=10)
sla_track_np = sla_track1.values
sla_track_np = fill_missing_nearest(sla_track_np)

# print(sla_track_np.shape)
# print(sla_track_np.count().
xout, yout, wout = loess_1d(
    gshhs,
    sla_track_np,
    xnew=None,
    degree=1,
    frac=0.1,
    npoints=None,
    rotate=False,
    sigy=None,
)
# print(x.shape)
# sla_track_np_lowess = lowess(sla_track_np, gshhs, frac=2.0 / 3, missing="none")
# sla_track_np_lowess = lowess(sla_track_np, x, frac=0.1, missing="none")
# plt.plot(gshhs, sla_track_np, label="raw")
plt.plot(gshhs, sla_track_np, label="raw")
# plt.plot(gshhs,sla_track_np_lowess[:, 1], label="lowess")
# plt.plot(gshhs, sla_track_np_lowess[:, 1], label="lowess")
plt.plot(xout, yout, label="lowess")
plt.legend()
info = ""
ax1.text(0.1, 0.02, info, transform=ax1.transAxes, fontsize=12, fontweight="bold")
plt.show()
sys.exit()
