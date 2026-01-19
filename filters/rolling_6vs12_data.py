import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs

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
sla_track_06 = sla_track.rolling(points_numbers=6, center=True).mean()
sla_track_12 = sla_track.rolling(points_numbers=12, center=True).mean()

corr_06 = xs.pearson_r(sla_track, sla_track_06, dim="points_numbers", skipna=True)
corr_12 = xs.pearson_r(sla_track, sla_track_12, dim="points_numbers", skipna=True)
rmse_06 = xs.rmse(sla_track, sla_track_06, dim="points_numbers", skipna=True)
rmse_12 = xs.rmse(sla_track, sla_track_12, dim="points_numbers", skipna=True)
info = (
    f"corr_06: {corr_06.item():.2f}, corr_12:{corr_12.item():.2f}, rmse_06:"
    f" {rmse_06.item():.2f}, rmse_12:{rmse_12.item():.2f}"
)

sla_track.plot(ax=ax1, label="no smoothing")
sla_track_06.plot(ax=ax1, label="06 point smoothing")
sla_track_12.plot(ax=ax1, label="12 point smoothing")
plt.legend()
ax1.text(0.1, 0.02, info, transform=ax1.transAxes, fontsize=12, fontweight="bold")
plt.show()
