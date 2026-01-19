import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs
import numpy as np
import statsmodels.api as sm
from loess.loess_1d import loess_1d
import sys

lowess = sm.nonparametric.lowess

# Define the window size for moving average
window_size = 12
# Generate the box signal of width 12
box_signal = np.ones(window_size) / window_size

sat = "S3A"
track_number = "195"
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

f_track = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number}.nc"
ds = xr.open_dataset(f_track, decode_times=False)
print(f_track)
print(ds)
cycle_len = len(ds.cycles_numbers)
lons_track = ds.lon.values
lats_track = ds.lat.values
x = ds.points_numbers.values
sla_track = ds.sla
sla_track = sla_track.isel(cycles_numbers=10)
sla_track_np = sla_track.values
print(sla_track_np.shape)
xout, yout, wout = loess_1d(
    x,
    sla_track_np,
    xnew=None,
    degree=1,
    frac=0.5,
    npoints=None,
    rotate=False,
    sigy=None,
)
print(x.shape)
sla_track_np_box = np.convolve(sla_track_np, box_signal, mode="same")
sla_track_np_rol = sla_track.rolling(points_numbers=12, center=True).mean()
sla_track_lowess = lowess(sla_track_np, x, frac=2.0 / 3, missing="none")
plt.plot(x, sla_track_np, label="raw")
plt.plot(x, sla_track_np_box, label="conv with box")
# plt.plot(x,sla_track_np_rol.values, label="rolling")
# plt.plot(x,sla_track_lowess[:, 1], label="lowess")
plt.plot(x_out, y_out, label="lowess")
plt.legend()
info = ""
ax1.text(0.1, 0.02, info, transform=ax1.transAxes, fontsize=12, fontweight="bold")
plt.show()
sys.exit()
