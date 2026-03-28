"""
Continuous Wavelet Transform (CWT) of RAMA SST at 0N, 67E (daily data).
Follows Torrence & Compo (1998) using the pycwt package.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find

# copilot --resume=c5161e73-97f4-4a7d-836b-d620eda107a1
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ds = xr.open_dataset('/home/srinivasu/allData/rama/sst/sst12n90e_dy.cdf')

# Squeeze out depth/lat/lon singleton dims → 1-D time series
sst = ds['T_25'].squeeze(drop=True)
sst = sst.sel(time=slice(None, '2020-12-31'))  # 20 years of data

# Convert time to decimal years for the wavelet routines
time_dt = sst['time'].values.astype('datetime64[D]')
t0_year = float(np.datetime64(time_dt[0], 'Y').astype(float)) + 1970
t_days  = (time_dt - time_dt[0]).astype(float)          # days from start
dt      = 1.0 / 365.25                                   # sampling: 1 day in years
t       = t_days * dt + t0_year                          # decimal years

# Raw SST values (NaN where missing)
dat = sst.values.astype(float)
dat[dat >= 1e34] = np.nan

# Interpolate over gaps so the FFT-based CWT receives a complete series
nans = np.isnan(dat)
if nans.any():
    idx = np.arange(len(dat))
    dat = np.interp(idx, idx[~nans], dat[~nans])

# ---------------------------------------------------------------------------
# Detrend & normalise
# ---------------------------------------------------------------------------
title  = "RAMA SST — 0°N, 67°E"
label  = "SST"
units  = "°C"

p            = np.polyfit(t - t[0], dat, 1)
dat_notrend  = dat - np.polyval(p, t - t[0])
std          = dat_notrend.std()
var          = std ** 2
dat_norm     = dat_notrend / std

# ---------------------------------------------------------------------------
# Wavelet parameters
# ---------------------------------------------------------------------------
mother = wavelet.Morlet(6)
s0     = 2 * dt                    # smallest scale  ≈ 2 days
dj     = 1 / 12                    # 12 sub-octaves per octave
J      = int(7 / dj)               # 7 powers of two → up to ~128 days; extend if needed
J      = int(np.log2((len(dat) * dt) / s0) / dj)  # span full data range
alpha, _, _ = wavelet.ar1(dat_norm) # lag-1 autocorrelation for red-noise background

# ---------------------------------------------------------------------------
# CWT & inverse CWT
# ---------------------------------------------------------------------------
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
    dat_norm, dt, dj, s0, J, mother
)
iwave  = wavelet.icwt(wave, scales, dt, dj, mother) * std

power      = np.abs(wave) ** 2
fft_power  = np.abs(fft) ** 2
period     = 1.0 / freqs           # in years

# Rectify bias (Liu et al. 2007)
power /= scales[:, None]

# ---------------------------------------------------------------------------
# Significance levels
# ---------------------------------------------------------------------------
signif, fft_theor = wavelet.significance(
    1.0, dt, scales, 0, alpha,
    significance_level=0.95, wavelet=mother
)
sig95 = (np.ones([1, len(dat)]) * signif[:, None])
sig95 = power / sig95              # ratio > 1 → significant

# Global wavelet spectrum and its significance
glbl_power = power.mean(axis=1)
dof        = len(dat) - scales
glbl_signif, _ = wavelet.significance(
    var, dt, scales, 1, alpha,
    significance_level=0.95, dof=dof, wavelet=mother
)

# Scale-averaged power: 0.5–2 year band
sel = find((period >= 0.5) & (period < 2))
Cdelta     = mother.cdelta
scale_avg  = (scales * np.ones((len(dat), 1))).T
scale_avg  = power / scale_avg
scale_avg  = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, _ = wavelet.significance(
    var, dt, scales, 2, alpha,
    significance_level=0.95,
    dof=[scales[sel[0]], scales[sel[-1]]],
    wavelet=mother
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.close('all')
plt.ioff()
fig = plt.figure(figsize=(11, 8), dpi=120)

# (a) Time series + inverse wavelet
ax = fig.add_axes([0.10, 0.75, 0.65, 0.20])
ax.plot(t, iwave, '-', linewidth=1, color='0.6', label='Inverse CWT')
ax.plot(t, dat_notrend, 'k', linewidth=1.2, label='SST anomaly')
ax.set_title(f'a) {title}')
ax.set_ylabel(f'{label} [{units}]')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([t.min(), t.max()])

# (b) Wavelet power spectrum
bx = fig.add_axes([0.10, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(
    t, np.log2(period), np.log2(power),
    np.log2(levels), extend='both', cmap='viridis'
)
bx.contour(
    t, np.log2(period), sig95, [-99, 1],
    colors='k', linewidths=1.5,
    extent=[t.min(), t.max(), 0, max(period)]
)
# Cone of influence
bx.fill(
    np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
    np.concatenate([
        np.log2(coi), [1e-9],
        np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]
    ]),
    'k', alpha=0.3, hatch='x'
)
bx.set_title(f'b) {label} Wavelet Power Spectrum ({mother.name})')
bx.set_ylabel('Period (years)')
Yticks = 2 ** np.arange(
    np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))
)
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels([f'{y:.3g}' for y in Yticks])

# (c) Global wavelet spectrum
cx = fig.add_axes([0.77, 0.37, 0.20, 0.28], sharey=bx)
cx.plot(glbl_signif,              np.log2(period), 'k--', label='95% signif.')
cx.plot(var * fft_theor,          np.log2(period), '--', color='#aaaaaa')
cx.plot(var * fft_power, np.log2(1.0 / fftfreqs), '-',  color='#aaaaaa', linewidth=1)
cx.plot(var * glbl_power,         np.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Spectrum')
cx.set_xlabel(f'Power [{units}²]')
cx.set_xlim([0, glbl_power.max() * var + var])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels([f'{y:.3g}' for y in Yticks])
plt.setp(cx.get_yticklabels(), visible=False)

# (d) Scale-averaged power (0.5–2 yr band)
dx = fig.add_axes([0.10, 0.07, 0.65, 0.20], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1)
dx.plot(t, scale_avg, 'k-', linewidth=1.2)
dx.set_title('d) 0.5–2 year scale-averaged power')
dx.set_xlabel('Time (year)')
dx.set_ylabel(f'Avg. variance [{units}²]')
ax.set_xlim([t.min(), t.max()])

out = 'cwt_sst_0n67e.png'
plt.savefig(out, bbox_inches='tight')
plt.show()
print(f'Saved → {out}')
