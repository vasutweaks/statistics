import gsw
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from tools_xtrack import *
import sys

DIST_THRESHOLD = 0.1
DIST_THRESHOLD_KM = 1.0
PIBY2 = np.pi / 2

import numpy as np


def genweights(p, q, dt=1):
    """
    Calculates optimal weighting coefficients and noise reduction factor for a digital filter.

    Args:
        p (int): Number of points before the point of interest (negative).
        q (int): Number of points after the point of interest (positive).
        dt (float, optional): Sampling period. Defaults to 1.

    Returns:
        tuple: (cn, error)
            - cn: NumPy array of optimal weighting coefficients.
            - error: Noise reduction factor of the filter.
    """
    # Input validation
    if not isinstance(p, int) or not isinstance(q, int):
        raise ValueError("p and q must be integers")
    if p >= 0 or q <= 0:
        raise ValueError("p must be negative and q must be positive")
    if p > -q:
        raise ValueError("p must be less than -q")
    # Total number of coefficients and matrix size
    print(p, q)
    N = abs(p) + abs(q)
    T = N + 1
    # Construct matrices A and B
    A = np.zeros((T, T))
    print(A.shape)
    A[-1, :-1] = 1  # Last row of A
    print(A)
    # n = np.arange(-p, q + 1)
    n = np.arange(p, q + 1)
    print(n)
    n = n[n != 0]
    print(n)
    for i in range(len(n)):
        A[i, :] = np.hstack([1.0 / n * (-n[i] / 2), n[i] ** 2 * dt**2 / 4])
        A[i, i] = -1
    B = np.zeros((T, 1))
    B[-1] = 1
    # Solve for coefficients and compute error
    cn = np.linalg.solve(A, B)[:-1]
    error = np.sqrt(np.sum(cn / (n * dt)) ** 2 + np.sum((cn / (n * dt)) ** 2))
    return cn, error


# Example usage
p = -6
q = 6
dt = 0.5
cn, error = genweights(p, q, dt)

print(f"Optimal weighting coefficients: {cn}")
print(f"Optimal weighting coefficients: {len(cn)}")
print(f"Optimal weighting coefficients: {cn.shape}")
print(f"Noise reduction factor: {error}")
cn1 = np.squeeze(cn)
weight = xr.DataArray(cn1, dims=["window"])

# plt.plot(cn)
# plt.show()
# sys.exit()

sat = "S3A"
track_number = "195"
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

tsta, tend = get_time_limits(sat)
track_tsta_o = datetime.strptime(tsta, "%Y-%m-%d")
track_tend_o = datetime.strptime(tend, "%Y-%m-%d")
f_track = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number}.nc"
ds = xr.open_dataset(f_track, decode_times=False)
print(f_track)
cycle_len = len(ds.cycles_numbers)
lons_track = ds.lon.values
lats_track = ds.lat.values
m = (lats_track[-1] - lats_track[0]) / (lons_track[-1] - lons_track[0])
angle_r = np.arctan(m)
angle = np.rad2deg(np.arctan(m))
angle_adj = 90.0 - abs(angle)
angle_adj_r = np.rad2deg(angle_adj)
print(
    track_number,
    sat,
    lons_track[0],
    lats_track[0],
    angle,
    "---------------------------",
)
# This is to identify the track point on which omni buoy falls
sla_track = track_dist_time_asn(ds, var_str="sla")
# sla_track = sla_track.resample(time="1M").mean()
sla_track = sla_track.isel(time=10)
sla_track.plot()
# print(sla_track)
# https://stackoverflow.com/questions/48510784/xarray-rolling-mean-with-weights/48512802#48512802
sla_track_1 = sla_track.rolling(x=12, center=True).mean()
sla_track_2 = sla_track.rolling(x=12, center=True).construct("window").dot(weight)

sla_track_1.plot()
sla_track_2.plot()
plt.show()
cn, error = genweights(-6, 5)

# break
