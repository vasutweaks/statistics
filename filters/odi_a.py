import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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
dt = 1
cn, error = genweights(p, q, dt)

print(f"Optimal weighting coefficients: {cn}")
print(f"Optimal weighting coefficients' shape: {cn.shape}")
print(f"Noise reduction factor: {error}")
cn1 = np.squeeze(cn)
# https://stackoverflow.com/questions/48510784/xarray-rolling-mean-with-weights/48512802#48512802
weights = xr.DataArray(cn1, dims=["window"])

sat = "S3A"
track_number = "195"
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

# plt.plot(cn1)
# plt.show()
# sys.exit()

f_track = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number}.nc"
ds = xr.open_dataset(f_track, decode_times=False)
print(f_track)
print(ds)
cycle_len = len(ds.cycles_numbers)
lons_track = ds.lon.values
lats_track = ds.lat.values
sla_track = ds.sla
sla_track = sla_track.isel(cycles_numbers=10)
sla_track_12 = sla_track.rolling(points_numbers=12, center=True).mean()
sla_track_12o = (
    sla_track.rolling(points_numbers=12, center=True).construct("window").dot(weights)
)

corr_12 = xr.corr(sla_track, sla_track_12, dim="points_numbers")
corr_12o = xr.corr(sla_track, sla_track_12o, dim="points_numbers")
info = f"corr_12: {corr_12.item():.2f}, corr_12o:{corr_12o.item():.2f}"

sla_track.plot(ax=ax1, label="no smoothing")
sla_track_12.plot(ax=ax1, label="12 point smoothing")
sla_track_12o.plot(ax=ax1, label="12 point smoothing with optimal weights")
plt.legend()
ax1.text(0.1, 0.01, info, transform=ax1.transAxes, fontsize=12, fontweight="bold")
plt.show()
