import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, freqz
import xarray as xr
from datetime import datetime

def print_theory(title, content):
    print("\n" + "="*80)
    print(f" THEORY: {title}")
    print("="*80)
    print(content)
    print("-"*80)
    input("Press Enter to continue to the demonstration...")

# --- EXAMPLE 1: Synthetic Data ---
theory_1 = """
Signal filtering is the process of removing unwanted components or features from a signal.
In the frequency domain, we define 'passbands' and 'stopbands'.

1. Low-pass Filter: Passes signals with a frequency lower than a selected cutoff frequency 
   and attenuates signals with frequencies higher than the cutoff.
2. Butterworth Filter: A type of signal processing filter designed to have a frequency 
   response as flat as possible in the passband. It is also referred to as a maximally 
   flat magnitude filter.
3. lfilter vs filtfilt:
   - lfilter: A causal filter that introduces a phase shift (delay) to the signal.
   - filtfilt: A zero-phase filter that processes the signal in both forward and backward 
     directions. This cancels the phase shift, resulting in zero phase distortion, but 
     it is non-causal (requires the whole signal beforehand).
"""
print_theory("Basics of Filtering & Zero-phase Distortion", theory_1)

# Generate synthetic data
fs = 1000       # Sample rate, Hz
T = 1.0         # Seconds
n = int(T * fs) # Total samples
t = np.linspace(0, T, n, endpoint=False)

# 5 Hz signal + 50 Hz noise
clean_signal = np.sin(2 * np.pi * 5 * t)
noise = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.random.randn(n)
dirty_signal = clean_signal + noise

# Filter parameters
cutoff = 15.0   # Cutoff frequency
order = 6       # Filter order

# Design Butterworth filter
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Apply filters
y_lfilter = lfilter(b, a, dirty_signal)
y_filtfilt = filtfilt(b, a, dirty_signal)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, dirty_signal, color='silver', label='Noisy signal')
plt.plot(t, clean_signal, 'k', label='Original 5Hz signal', linewidth=2)
plt.plot(t, y_lfilter, 'r', label='lfilter (Phase shift!)')
plt.plot(t, y_filtfilt, 'g', label='filtfilt (Zero-phase)')
plt.title("Synthetic Data: Low-pass Butterworth Filtering")
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

input("\nExample 1 Finished. Press Enter to proceed to Real Data Example...")

# --- EXAMPLE 2: Real Data (SST) ---
theory_2 = """
Real-world data often contains multiple scales of variability. 
Sea Surface Temperature (SST) data from RAMA buoys typically shows:
- Annual cycle (seasonal variations)
- Interannual variability (like El Niño)
- High-frequency noise (diurnal cycles, sensor noise, short-term weather)

We will use:
1. Low-pass filter to extract the smooth seasonal/interannual trend.
2. High-pass filter to isolate the high-frequency 'anomalies' or noise.
3. Band-pass filter to isolate a specific frequency range.
"""
print_theory("Filtering Real Environmental Time Series", theory_2)

data_path = '/home/srinivasu/allData/rama/sst/sst0n90e_dy.cdf'
ds = xr.open_dataset(data_path)
# Extract SST and handle missing values (often 1e35 in NetCDF)
# xarray usually masks missing_value attributes as NaN, 
# but let's be explicit to ensure values > 100 (like 1e35) are NaN.
sst_raw = ds.T_25.isel(depth=0, lon=0, lat=0, drop=True).squeeze()
sst_raw = sst_raw.where(sst_raw < 100)
# Use xarray's native interpolation for missing values to avoid filter artifacts
# This is more idiomatic than using scipy.interpolate manually
sst_clean_xr = sst_raw.interpolate_na(dim="time", method="linear")

# Convert to numpy for filtering with scipy.signal
sst_clean = sst_clean_xr.values

fs_daily = 1.0 # 1 sample per day

# 1. Low-pass: Keep periods > 90 days (seasonal trend)
cutoff_low = 1.0/90.0 # Frequency in 1/days
b_low, a_low = butter(4, cutoff_low / (0.5 * fs_daily), btype='low')
sst_trend = filtfilt(b_low, a_low, sst_clean)

# 2. High-pass: Keep periods < 10 days (high-frequency noise)
cutoff_high = 1.0/10.0
b_high, a_high = butter(4, cutoff_high / (0.5 * fs_daily), btype='high')
sst_noise = filtfilt(b_high, a_high, sst_clean)

    # Plotting Real Data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(ds.time, sst_clean, color='gray', alpha=0.5, label='Interpolated SST (xarray)')
ax1.plot(ds.time, sst_trend, color='red', linewidth=2, label='Seasonal Trend (>90 days)')
ax1.set_title("RAMA SST (0N, 90E): Low-pass Filtering")
ax1.legend()
ax1.grid(True)

ax2.plot(ds.time, sst_noise, color='blue', label='High-freq components (<10 days)')
ax2.set_title("High-pass Filtering (Anomalies/Noise)")
ax2.legend()
ax2.grid(True)

# 3. Band-pass: 30-60 day variations (e.g., Madden-Julian Oscillation scale)
low_bp = 1.0/60.0
high_bp = 1.0/30.0
b_band, a_band = butter(4, [low_bp / (0.5 * fs_daily), high_bp / (0.5 * fs_daily)], btype='band')
sst_band = filtfilt(b_band, a_band, sst_clean)

ax3.plot(ds.time, sst_band, color='green', label='Intraseasonal (30-60 days)')
ax3.set_title("Band-pass Filtering (Intraseasonal scales)")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print(" DEMO COMPLETE")
print("="*80)
