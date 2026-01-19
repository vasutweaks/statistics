import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Designs and applies a Butterworth low-pass filter.
    Parameters:
    - data: 1D numpy array, the time series data.
    - cutoff: float, the cutoff frequency in cycles per day.
    - fs: float, the sampling frequency in samples per day.
    - order: int, the order of the filter.
    Returns:
    - y: 1D numpy array, the filtered data.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def filter_time_series(da, cutoff_days=3, order=4):
    """
    Applies a Butterworth low-pass filter along the 'time' dimension.
    Parameters:
    - da: xarray DataArray, input data with dimensions ('x', 'time').
    - cutoff_days: float, the cutoff period in days.
    - order: int, the order of the Butterworth filter.
    Returns:
    - filtered_da: xarray DataArray, the filtered data.
    """
    fs = 1  # Sampling frequency: 1 sample per day
    cutoff = 1 / cutoff_days  # Cutoff frequency in cycles per day

    def filter_func(ts):
        """
        Filters a 1D time series, handling missing values.
        Parameters:
        - ts: 1D numpy array, the time series data.
        Returns:
        - y: 1D numpy array, the filtered time series.
        """
        # Find indices where data is not NaN
        not_nan = ~np.isnan(ts)
        if np.sum(not_nan) < 2:
            # Not enough data to interpolate
            return ts
        else:
            # Interpolate missing values
            x = np.arange(len(ts))
            ts_interp = np.interp(x, x[not_nan], ts[not_nan])
            # Apply the Butterworth filter
            y = butter_lowpass_filter(ts_interp, cutoff, fs, order)
            # Restore NaN values in their original positions
            y[~not_nan] = np.nan
            return y

    # Apply the filter function along the 'time' dimension
    filtered_da = xr.apply_ufunc(
        filter_func,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',  # If using Dask for parallel computation
        output_dtypes=[da.dtype]
    )
    return filtered_da

# Usage:
# Assuming 'dataarray' is your input DataArray with dimensions ('x', 'time')
smoothed_da = filter_time_series(dataarray, cutoff_days=3, order=4)

WBOB = (78.0, 87.0, 8.0, 21.0)
TN_RADAR = (79.6, 81.8, 10.5, 12.9)

dsr = xr.open_dataset("/home/srinivasu/allData/radar/TNCodar_daily.nc")
print(dsr)
dsr = dsr.rename(
    {"XAXS": "longitude", "YAXS": "latitude", "ZAXS": "lev", "TAXIS1D": "time"}
)
print(dsr)
u_radar = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
u_radar_m = u_radar.resample(time="1M").mean()
v_radar = 0.01 * dsr.V_RADAR.isel(lev=0, drop=True)
v_radar_m = v_radar.resample(time="1M").mean()
print(u_radar)

sizes = u_radar.sizes
ln = sizes["time"]

midpoint = ((TN_RADAR[0] + TN_RADAR[1]) / 2, (TN_RADAR[2] + TN_RADAR[3]) / 2)
u_radar1 = u_radar.sel(longitude=midpoint[0], latitude=midpoint[1], method="nearest")
u_radar_np = u_radar1.values

not_nan = ~np.isnan(u_radar_np)
if np.sum(not_nan) < 2:
    # Not enough data to interpolate
    return ts
else:
    # Interpolate missing values
    x = np.arange(len(u_radar_np))
    ts_interp = np.interp(x, x[not_nan], u_radar_np[not_nan])
    # Apply the Butterworth filter
    y = butter_lowpass_filter(ts_interp, cutoff, fs, order)
    # Restore NaN values in their original positions
    y[~not_nan] = np.nan
    return y
fig, ax = plt.subplots(1,1, figsize=(8,4), layout="constrained")
fs = 1
n = 3 # 8 points is around 2 kilometers  (1 point every 300 meters approx)
cutoff = 1/n

u_radar_np1 = butter_lowpass_filter(u_radar_np, cutoff, fs, order=5)

plt.plot(u_radar_np[4500:5000], label = "before filter")
plt.plot(u_radar_np1[4500:5000], label = "after filter")
plt.legend()
plt.show()
