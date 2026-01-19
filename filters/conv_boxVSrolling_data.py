import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs
import numpy as np
import sys

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
sla_track = ds.sla
sla_track = sla_track.isel(cycles_numbers=10)
sla_track_np = sla_track.values
print(sla_track_np.shape)

x = ds.points_numbers.values
sla_track_np_box = np.convolve(sla_track_np, box_signal, mode="same")
sla_track_np_rol = sla_track.rolling(points_numbers=12, center=True).mean()
plt.plot(x, sla_track_np, label="raw")
plt.plot(x, sla_track_np_box, label="conv with box")
plt.plot(x, sla_track_np_rol.values, "+b", label="rolling")
plt.legend()
info = ""
ax1.text(0.1, 0.02, info, transform=ax1.transAxes, fontsize=12, fontweight="bold")
plt.show()
sys.exit()
