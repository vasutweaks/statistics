#!/usr/bin/env python3
"""
Lead-Lag Correlation and Decorrelation Length Analysis for Along-Track X-TRACK SLA
Missions: Sentinel-3A (Track 066, Western Bay of Bengal) and GFO (Track 008, Andaman Sea)

References:
- Draft manuscript (Section 2: Data and Methods): /home/srinivasu/slnew/coastal_alt_docs/drafts/draft_v02.2a.pdf
- Library tools: /home/srinivasu/xtrackm/lib/tools_xtrackm.py
- Romanou et al. (2006), J. Climate (Decorrelation scale definitions)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Add local tools library to path
sys.path.append("/home/srinivasu/xtrackm/lib")
import tools_xtrackm as tx

# Output directory for plots and tables
OUT_DIR = Path("/home/srinivasu/statistics/decorrelation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define dataset paths
XTRACK_S3A_FILE = Path("/home/srinivasu/xtrackm/data/S3A/ctoh.sla.ref.S3A.nindian.066.nc")
XTRACK_GFO_FILE = Path("/home/srinivasu/xtrackm/data/GFO/ctoh.sla.ref.GFO.nindian.008.nc")


# ---------------------------------------------------------
# Fitting functions for decorrelation scale estimation
# ---------------------------------------------------------
def exp_decay(r, L):
    """Exponential decay: R(r) = exp(-r / L)"""
    return np.exp(-r / L)


def gauss_decay(r, L):
    """Gaussian decay: R(r) = exp(-(r / L)^2)"""
    return np.exp(-(r / L) ** 2)


def exp_cos_decay(r, L, k):
    """Exponentially damped cosine decay: R(r) = exp(-r / L) * cos(k * r)"""
    return np.exp(-r / L) * np.cos(k * r)


# ---------------------------------------------------------
# Load and preprocess X-TRACK SLA dataset
# ---------------------------------------------------------
def load_and_preprocess_xtrack(filepath, sat_name, apply_mad=True, mad_thresh=3.5):
    """
    Loads along-track SLA, extracts along-track distance x (km from coast),
    lons, lats, and time, and optionally applies MAD outlier filtering.
    """
    print(f"\nLoading {sat_name} from: {filepath}")
    ds = xr.open_dataset(filepath, decode_times=False)
    
    # Extract SLA with distance from coast in km
    sla_da = tx.track_dist_time_asn(ds, var_str="sla", units_in="km")
    
    # Store track coordinates
    lons_rev = ds.lon.values[::-1]
    lats_rev = ds.lat.values[::-1]
    dist_gshhg_rev = ds.dist_to_coast_gshhg.values[::-1] / 1000.0  # in km
    
    # Attach coordinates
    sla_da = sla_da.assign_coords(
        lon=("x", lons_rev),
        lat=("x", lats_rev),
        dist_gshhg=("x", dist_gshhg_rev),
    )
    
    # Apply MAD outlier filter if requested
    if apply_mad:
        print(f"Applying MAD outlier filter (threshold = {mad_thresh})...")
        sla_filtered = tx.mad_filter_2d_xr(sla_da, thresh=mad_thresh, dims=("x", "time"))
    else:
        sla_filtered = sla_da
        
    return ds, sla_filtered


# ---------------------------------------------------------
# Compute Spatial Autocorrelation Function (ACF)
# ---------------------------------------------------------
def compute_spatial_acf(da, max_lag_km=1200.0, bin_width_km=7.0):
    """
    Computes along-track spatial autocorrelation function R(Delta x) by
    calculating pairwise cross-correlations between all spatial points along the track.
    """
    x = da.x.values
    x_size = len(x)
    
    # Temporal anomalies (relative to point-wise temporal mean)
    da_anom = (da - da.mean(dim="time", skipna=True)).values
    
    pair_dists = []
    pair_corrs = []
    
    for i in range(x_size):
        ts_i = da_anom[i, :]
        mask_i = ~np.isnan(ts_i)
        if np.sum(mask_i) < 8:
            continue
        var_i = np.nanvar(ts_i)
        if var_i <= 1e-8 or np.isnan(var_i):
            continue
            
        for j in range(i, x_size):
            ts_j = da_anom[j, :]
            mask_j = ~np.isnan(ts_j)
            valid = mask_i & mask_j
            if np.sum(valid) < 8:
                continue
            var_j = np.nanvar(ts_j)
            if var_j <= 1e-8 or np.isnan(var_j):
                continue
                
            r = np.corrcoef(ts_i[valid], ts_j[valid])[0, 1]
            if not np.isnan(r):
                d = abs(x[i] - x[j])
                pair_dists.append(d)
                pair_corrs.append(r)
                
    pair_dists = np.array(pair_dists)
    pair_corrs = np.array(pair_corrs)
    
    # Bin correlations by distance separation
    bins = np.arange(0, max_lag_km + bin_width_km, bin_width_km)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    acf_mean = []
    acf_std = []
    acf_count = []
    
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        cond = (pair_dists >= b_low) & (pair_dists < b_high)
        if np.sum(cond) > 0:
            acf_mean.append(np.nanmean(pair_corrs[cond]))
            acf_std.append(np.nanstd(pair_corrs[cond]))
            acf_count.append(np.sum(cond))
        else:
            acf_mean.append(np.nan)
            acf_std.append(np.nan)
            acf_count.append(0)
            
    return (
        bin_centers,
        np.array(acf_mean),
        np.array(acf_std),
        np.array(acf_count),
        pair_dists,
        pair_corrs,
    )


# ---------------------------------------------------------
# Estimate Decorrelation Scales from Spatial ACF
# ---------------------------------------------------------
def estimate_decorrelation_scales(lags, acf, fit_max_km=600.0):
    """
    Estimates decorrelation scales using:
    1. e-folding scale (R = 1/e ≈ 0.368)
    2. First zero-crossing scale (R = 0, Romanou et al. 2006)
    3. Exponential fit: R(r) = exp(-r / L_exp)
    4. Gaussian fit: R(r) = exp(-(r / L_gauss)^2)
    5. Integral length scale: integral of R(r) dr from 0 to L_zero
    """
    valid = ~np.isnan(acf)
    lags_v = lags[valid]
    acf_v = acf[valid]
    
    # 1. e-folding scale (interpolation for sub-grid precision)
    target_efold = 1.0 / np.e
    efold_scale = np.nan
    if np.any(acf_v <= target_efold):
        idx_e = np.where(acf_v <= target_efold)[0][0]
        if idx_e > 0:
            # Linear interpolation between points
            r1, r2 = acf_v[idx_e - 1], acf_v[idx_e]
            d1, d2 = lags_v[idx_e - 1], lags_v[idx_e]
            efold_scale = d1 + (target_efold - r1) * (d2 - d1) / (r2 - r1)
        else:
            efold_scale = lags_v[0]
            
    # 2. First zero-crossing scale (Romanou et al., 2006)
    zero_crossing = np.nan
    if np.any(acf_v < 0):
        idx_z = np.where(acf_v < 0)[0][0]
        if idx_z > 0:
            r1, r2 = acf_v[idx_z - 1], acf_v[idx_z]
            d1, d2 = lags_v[idx_z - 1], lags_v[idx_z]
            zero_crossing = d1 + (0.0 - r1) * (d2 - d1) / (r2 - r1)
        else:
            zero_crossing = lags_v[0]
            
    # 3. Exponential curve fit
    fit_mask = valid & (lags <= fit_max_km)
    try:
        popt_exp, _ = curve_fit(exp_decay, lags[fit_mask], acf[fit_mask], p0=[150.0], bounds=(1.0, 5000.0))
        L_exp = popt_exp[0]
    except Exception as e:
        L_exp = np.nan
        
    # 4. Gaussian curve fit
    try:
        popt_gauss, _ = curve_fit(gauss_decay, lags[fit_mask], acf[fit_mask], p0=[150.0], bounds=(1.0, 5000.0))
        L_gauss = popt_gauss[0]
    except Exception as e:
        L_gauss = np.nan
        
    # 5. Integral scale (trapezoidal integration up to zero crossing or max range)
    upper_limit = zero_crossing if not np.isnan(zero_crossing) else (efold_scale if not np.isnan(efold_scale) else lags_v[-1])
    int_mask = valid & (lags <= upper_limit) & (acf >= 0)
    if np.sum(int_mask) > 2:
        L_integral = np.trapezoid(acf[int_mask], lags[int_mask])
    else:
        L_integral = np.nan
        
    return {
        "e_folding_km": efold_scale,
        "zero_crossing_km": zero_crossing,
        "exponential_fit_km": L_exp,
        "gaussian_fit_km": L_gauss,
        "integral_scale_km": L_integral,
    }


# ---------------------------------------------------------
# Compute Local Spatial Decorrelation Scale along Track
# ---------------------------------------------------------
def compute_local_spatial_decorrelation(da, target_r=1.0/np.e):
    """
    Computes local decorrelation length L_d(x) for each spatial point x along the track.
    """
    x = da.x.values
    x_size = len(x)
    da_anom = (da - da.mean(dim="time", skipna=True)).values
    
    local_Ld = np.full(x_size, np.nan)
    
    for i in range(x_size):
        ts_i = da_anom[i, :]
        mask_i = ~np.isnan(ts_i)
        if np.sum(mask_i) < 8:
            continue
            
        corrs = []
        dists = []
        for j in range(x_size):
            ts_j = da_anom[j, :]
            mask_j = ~np.isnan(ts_j)
            valid = mask_i & mask_j
            if np.sum(valid) >= 8:
                r = np.corrcoef(ts_i[valid], ts_j[valid])[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
                    dists.append(abs(x[i] - x[j]))
                    
        dists = np.array(dists)
        corrs = np.array(corrs)
        
        # Sort by distance
        sort_idx = np.argsort(dists)
        dists_s = dists[sort_idx]
        corrs_s = corrs[sort_idx]
        
        # Bin or smooth by distance
        b_edges = np.arange(0, 800, 10.0)
        b_c = 0.5 * (b_edges[:-1] + b_edges[1:])
        b_r = []
        for b0, b1 in zip(b_edges[:-1], b_edges[1:]):
            c = (dists_s >= b0) & (dists_s < b1)
            b_r.append(np.nanmean(corrs_s[c]) if np.sum(c) > 0 else np.nan)
        b_r = np.array(b_r)
        
        # Find e-folding scale
        valid_b = ~np.isnan(b_r)
        if np.any(b_r[valid_b] <= target_r):
            idx_e = np.where(b_r[valid_b] <= target_r)[0][0]
            b_c_v = b_c[valid_b]
            b_r_v = b_r[valid_b]
            if idx_e > 0:
                r1, r2 = b_r_v[idx_e - 1], b_r_v[idx_e]
                d1, d2 = b_c_v[idx_e - 1], b_c_v[idx_e]
                local_Ld[i] = d1 + (target_r - r1) * (d2 - d1) / (r2 - r1)
            else:
                local_Ld[i] = b_c_v[0]
                
    return local_Ld


# ---------------------------------------------------------
# Spatiotemporal Lead-Lag Cross-Correlation Analysis
# ---------------------------------------------------------
def compute_lead_lag_matrix(da, ref_idx, max_tau_cycles=15, dt_days=27.0):
    """
    Computes lead-lag cross-correlation R(x, tau) between reference point
    and all points x along the track for time lags tau in [-max_tau, +max_tau].
    
    Convention:
    tau > 0: point x LAGS reference point by tau (reference leads, signal moves to x)
    tau < 0: point x LEADS reference point by |tau|
    """
    x = da.x.values
    x_size = len(x)
    t_size = da.time.size
    da_anom = (da - da.mean(dim="time", skipna=True)).values
    
    tau_lags = np.arange(-max_tau_cycles, max_tau_cycles + 1)
    ll_matrix = np.full((x_size, len(tau_lags)), np.nan)
    
    ts_ref = da_anom[ref_idx, :]
    
    for i in range(x_size):
        ts_i = da_anom[i, :]
        for k, tau in enumerate(tau_lags):
            if tau == 0:
                valid = ~np.isnan(ts_ref) & ~np.isnan(ts_i)
                if np.sum(valid) >= 6:
                    ll_matrix[i, k] = np.corrcoef(ts_ref[valid], ts_i[valid])[0, 1]
            elif tau > 0:  # ts_i at t+tau, ts_ref at t
                s_ref = ts_ref[:-tau]
                s_i = ts_i[tau:]
                valid = ~np.isnan(s_ref) & ~np.isnan(s_i)
                if np.sum(valid) >= 6:
                    ll_matrix[i, k] = np.corrcoef(s_ref[valid], s_i[valid])[0, 1]
            else:  # tau < 0: ts_i at t-|tau|, ts_ref at t
                abs_tau = abs(tau)
                s_ref = ts_ref[abs_tau:]
                s_i = ts_i[:-abs_tau]
                valid = ~np.isnan(s_ref) & ~np.isnan(s_i)
                if np.sum(valid) >= 6:
                    ll_matrix[i, k] = np.corrcoef(s_ref[valid], s_i[valid])[0, 1]
                    
    return tau_lags, tau_lags * dt_days, ll_matrix


# ---------------------------------------------------------
# Compute Temporal Autocorrelation along Track
# ---------------------------------------------------------
def compute_temporal_acf_along_track(da, max_tau_cycles=15, dt_days=27.0):
    """
    Computes temporal autocorrelation R_t(tau) for each point along the track.
    """
    x = da.x.values
    x_size = len(x)
    da_anom = (da - da.mean(dim="time", skipna=True)).values
    
    tau_lags = np.arange(0, max_tau_cycles + 1)
    temp_acf = np.full((x_size, len(tau_lags)), np.nan)
    
    for i in range(x_size):
        ts = da_anom[i, :]
        for k, tau in enumerate(tau_lags):
            if tau == 0:
                valid = ~np.isnan(ts)
                temp_acf[i, k] = 1.0 if np.sum(valid) >= 6 else np.nan
            else:
                s1 = ts[:-tau]
                s2 = ts[tau:]
                valid = ~np.isnan(s1) & ~np.isnan(s2)
                if np.sum(valid) >= 6:
                    temp_acf[i, k] = np.corrcoef(s1[valid], s2[valid])[0, 1]
                    
    return tau_lags, tau_lags * dt_days, temp_acf


# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def run_full_analysis():
    print("=" * 70)
    print("STARTING ALONG-TRACK SLA LEAD-LAG & DECORRELATION ANALYSIS")
    print("=" * 70)
    
    # 1. Load Datasets
    ds_s3a, sla_s3a = load_and_preprocess_xtrack(XTRACK_S3A_FILE, "Sentinel-3A Pass 066")
    ds_gfo, sla_gfo = load_and_preprocess_xtrack(XTRACK_GFO_FILE, "GFO Pass 008")
    
    dt_s3a = 27.0  # days
    dt_gfo = 17.0  # days
    
    # Deseasonalized (non-seasonal anomaly)
    def deseasonalize(da):
        anom = (da - da.mean(dim="time", skipna=True)).values
        times = pd.to_datetime(da.time.values)
        months = times.month
        out = np.zeros_like(anom)
        for m in range(1, 13):
            idx = (months == m)
            if np.any(idx):
                clim = np.nanmean(anom[:, idx], axis=1)
                out[:, idx] = anom[:, idx] - clim[:, np.newaxis]
        return xr.DataArray(out, coords=da.coords, dims=da.dims)
        
    sla_s3a_ns = deseasonalize(sla_s3a)
    sla_gfo_ns = deseasonalize(sla_gfo)
    
    # 2. Compute Spatial ACF and Decorrelation Scales
    print("\n--- Computing Spatial Autocorrelations ---")
    lags_s3a, acf_s3a_tot, _, _, _, _ = compute_spatial_acf(sla_s3a, max_lag_km=1400.0)
    lags_s3a_ns, acf_s3a_ns, _, _, _, _ = compute_spatial_acf(sla_s3a_ns, max_lag_km=1400.0)
    
    lags_gfo, acf_gfo_tot, _, _, _, _ = compute_spatial_acf(sla_gfo, max_lag_km=1400.0)
    lags_gfo_ns, acf_gfo_ns, _, _, _, _ = compute_spatial_acf(sla_gfo_ns, max_lag_km=1400.0)
    
    # Estimate scales
    scales_s3a_tot = estimate_decorrelation_scales(lags_s3a, acf_s3a_tot, fit_max_km=700.0)
    scales_s3a_ns = estimate_decorrelation_scales(lags_s3a_ns, acf_s3a_ns, fit_max_km=700.0)
    scales_gfo_tot = estimate_decorrelation_scales(lags_gfo, acf_gfo_tot, fit_max_km=700.0)
    scales_gfo_ns = estimate_decorrelation_scales(lags_gfo_ns, acf_gfo_ns, fit_max_km=700.0)
    
    # Local decorrelation scale along track
    print("--- Computing Local Along-Track Decorrelation Lengths ---")
    local_ld_s3a = compute_local_spatial_decorrelation(sla_s3a)
    local_ld_gfo = compute_local_spatial_decorrelation(sla_gfo)
    
    # 3. Spatiotemporal Lead-Lag Correlation Relative to Coast
    print("--- Computing Lead-Lag Correlation Relative to Near-Coast ---")
    idx_coast_s3a = 0  # x = 0 km (closest to coast)
    idx_50km_s3a = np.argmin(np.abs(sla_s3a.x.values - 50.0))
    idx_coast_gfo = 0
    idx_50km_gfo = np.argmin(np.abs(sla_gfo.x.values - 50.0))
    
    tau_s3a, tau_days_s3a, ll_s3a_coast = compute_lead_lag_matrix(sla_s3a, idx_coast_s3a, max_tau_cycles=14, dt_days=dt_s3a)
    _, _, ll_s3a_50km = compute_lead_lag_matrix(sla_s3a, idx_50km_s3a, max_tau_cycles=14, dt_days=dt_s3a)
    _, _, ll_s3a_ns_50km = compute_lead_lag_matrix(sla_s3a_ns, idx_50km_s3a, max_tau_cycles=14, dt_days=dt_s3a)
    
    tau_gfo, tau_days_gfo, ll_gfo_coast = compute_lead_lag_matrix(sla_gfo, idx_coast_gfo, max_tau_cycles=15, dt_days=dt_gfo)
    _, _, ll_gfo_50km = compute_lead_lag_matrix(sla_gfo, idx_50km_gfo, max_tau_cycles=15, dt_days=dt_gfo)
    
    # 4. Temporal Autocorrelation Along Track
    print("--- Computing Temporal Autocorrelations Along Track ---")
    tau_t_s3a, t_days_s3a, temp_acf_s3a = compute_temporal_acf_along_track(sla_s3a, max_tau_cycles=14, dt_days=dt_s3a)
    tau_t_gfo, t_days_gfo, temp_acf_gfo = compute_temporal_acf_along_track(sla_gfo, max_tau_cycles=15, dt_days=dt_gfo)
    
    # Summary Metrics Table
    metrics_summary = [
        {
            "Mission": "Sentinel-3A Pass 066",
            "Region": "Western Bay of Bengal (EICC)",
            "Signal": "Total SLA",
            "e_folding_Ld_km": scales_s3a_tot["e_folding_km"],
            "zero_crossing_Ld_km": scales_s3a_tot["zero_crossing_km"],
            "exponential_fit_Ld_km": scales_s3a_tot["exponential_fit_km"],
            "gaussian_fit_Ld_km": scales_s3a_tot["gaussian_fit_km"],
            "integral_scale_km": scales_s3a_tot["integral_scale_km"],
        },
        {
            "Mission": "Sentinel-3A Pass 066",
            "Region": "Western Bay of Bengal (EICC)",
            "Signal": "Deseasonalized SLA",
            "e_folding_Ld_km": scales_s3a_ns["e_folding_km"],
            "zero_crossing_Ld_km": scales_s3a_ns["zero_crossing_km"],
            "exponential_fit_Ld_km": scales_s3a_ns["exponential_fit_km"],
            "gaussian_fit_Ld_km": scales_s3a_ns["gaussian_fit_km"],
            "integral_scale_km": scales_s3a_ns["integral_scale_km"],
        },
        {
            "Mission": "GFO Pass 008",
            "Region": "Andaman Sea / Eastern BoB",
            "Signal": "Total SLA",
            "e_folding_Ld_km": scales_gfo_tot["e_folding_km"],
            "zero_crossing_Ld_km": scales_gfo_tot["zero_crossing_km"],
            "exponential_fit_Ld_km": scales_gfo_tot["exponential_fit_km"],
            "gaussian_fit_Ld_km": scales_gfo_tot["gaussian_fit_km"],
            "integral_scale_km": scales_gfo_tot["integral_scale_km"],
        },
        {
            "Mission": "GFO Pass 008",
            "Region": "Andaman Sea / Eastern BoB",
            "Signal": "Deseasonalized SLA",
            "e_folding_Ld_km": scales_gfo_ns["e_folding_km"],
            "zero_crossing_Ld_km": scales_gfo_ns["zero_crossing_km"],
            "exponential_fit_Ld_km": scales_gfo_ns["exponential_fit_km"],
            "gaussian_fit_Ld_km": scales_gfo_ns["gaussian_fit_km"],
            "integral_scale_km": scales_gfo_ns["integral_scale_km"],
        },
    ]
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(OUT_DIR / "decorrelation_metrics_summary.csv", index=False)
    print("\n=== DECORRELATION METRICS SUMMARY ===")
    print(df_metrics.to_string())
    
    # ---------------------------------------------------------
    # GENERATE PUBLICATION-QUALITY FIGURES
    # ---------------------------------------------------------
    print("\n--- Generating Figures ---")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})
    
    # ---------------------------------------------------------
    # FIGURE 1: Geographic Track Map with GSHHG Distance to Coast
    # ---------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 6))
    ax_map = fig1.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax_map.set_extent([75, 102, -2, 22], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor="#e0e0e0", zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, edgecolor="k", linewidth=1.2, zorder=2)
    ax_map.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="#666666", zorder=2)
    
    sc1 = ax_map.scatter(
        sla_s3a.lon.values,
        sla_s3a.lat.values,
        c=sla_s3a.x.values,
        cmap="viridis",
        s=12,
        label="S3A Pass 066",
        zorder=3,
        transform=ccrs.PlateCarree(),
    )
    sc2 = ax_map.scatter(
        sla_gfo.lon.values,
        sla_gfo.lat.values,
        c=sla_gfo.x.values,
        cmap="plasma",
        s=12,
        label="GFO Pass 008",
        zorder=3,
        transform=ccrs.PlateCarree(),
    )
    
    # Mark coastal termination points
    ax_map.plot(sla_s3a.lon.values[0], sla_s3a.lat.values[0], "r*", markersize=12, label="S3A Coastal Point (x=0km)", zorder=4)
    ax_map.plot(sla_gfo.lon.values[0], sla_gfo.lat.values[0], "b*", markersize=12, label="GFO Coastal Point (x=0km)", zorder=4)
    
    ax_map.set_title("(a) Ground Track Trajectories & Orientation")
    ax_map.set_xticks(range(75, 105, 5), crs=ccrs.PlateCarree())
    ax_map.set_yticks(range(0, 25, 5), crs=ccrs.PlateCarree())
    ax_map.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°E"))
    ax_map.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°N"))
    ax_map.legend(loc="lower left", fontsize=9)
    ax_map.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot (b): Distance to Coast Profile
    ax_dist = fig1.add_subplot(1, 2, 2)
    ax_dist.plot(sla_s3a.x.values, sla_s3a.dist_gshhg.values, "g-", lw=2, label="S3A 066 (Western BoB)")
    ax_dist.plot(sla_gfo.x.values, sla_gfo.dist_gshhg.values, "m--", lw=2, label="GFO 008 (Andaman Sea)")
    ax_dist.axhline(50, color="gray", linestyle=":", label="50 km Coastal Boundary")
    ax_dist.set_xlabel("Along-Track Distance from Terminal Point $s$ (km)")
    ax_dist.set_ylabel("True Distance to Coast (GSHHG, km)")
    ax_dist.set_title("(b) Along-Track Distance vs. Coast Proximity")
    ax_dist.legend(loc="upper left")
    ax_dist.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig1.savefig(OUT_DIR / "fig1_track_geography.png", dpi=300)
    plt.close(fig1)
    
    # ---------------------------------------------------------
    # FIGURE 2: Spatiotemporal SLA Hovmöller Diagrams
    # ---------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    
    # S3A Hovmöller
    times_s3a_dt = pd.to_datetime(sla_s3a.time.values)
    c1 = ax2a.pcolormesh(
        times_s3a_dt,
        sla_s3a.x.values,
        sla_s3a.values,
        cmap="RdBu_r",
        vmin=-0.25,
        vmax=0.25,
        shading="auto",
    )
    ax2a.set_ylabel("Along-Track Distance (km)\n[0 = Andhra Coast, ~2000 = Equator]")
    ax2a.set_title("(a) Sentinel-3A Pass 066 Along-Track SLA (2016–2024, Western Bay of Bengal)")
    fig2.colorbar(c1, ax=ax2a, label="SLA (m)", pad=0.01)
    ax2a.grid(True, linestyle=":", alpha=0.4)
    
    # GFO Hovmöller
    times_gfo_dt = pd.to_datetime(sla_gfo.time.values)
    c2 = ax2b.pcolormesh(
        times_gfo_dt,
        sla_gfo.x.values,
        sla_gfo.values,
        cmap="RdBu_r",
        vmin=-0.25,
        vmax=0.25,
        shading="auto",
    )
    ax2b.set_xlabel("Time (Year)")
    ax2b.set_ylabel("Along-Track Distance (km)\n[0 = Andaman Coast, ~1500 = Equator]")
    ax2b.set_title("(b) GFO Pass 008 Along-Track SLA (2000–2008, Andaman Sea / Eastern Bay of Bengal)")
    fig2.colorbar(c2, ax=ax2b, label="SLA (m)", pad=0.01)
    ax2b.grid(True, linestyle=":", alpha=0.4)
    
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "fig2_hovmoller_sla.png", dpi=300)
    plt.close(fig2)
    
    # ---------------------------------------------------------
    # FIGURE 3: Along-Track Spatial Cross-Correlation Matrices
    # ---------------------------------------------------------
    def compute_corr_matrix(da):
        anom = (da - da.mean(dim="time", skipna=True)).values
        x_sz = da.x.size
        cmat = np.full((x_sz, x_sz), np.nan)
        for i in range(x_sz):
            for j in range(x_sz):
                t1, t2 = anom[i, :], anom[j, :]
                v = ~np.isnan(t1) & ~np.isnan(t2)
                if np.sum(v) >= 8:
                    cmat[i, j] = np.corrcoef(t1[v], t2[v])[0, 1]
        return cmat

    cmat_s3a = compute_corr_matrix(sla_s3a)
    cmat_gfo = compute_corr_matrix(sla_gfo)
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax3a.pcolormesh(
        sla_s3a.x.values,
        sla_s3a.x.values,
        cmat_s3a,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=1.0,
        shading="auto",
    )
    ax3a.set_xlabel("Along-Track Distance $s_1$ (km from Coast)")
    ax3a.set_ylabel("Along-Track Distance $s_2$ (km from Coast)")
    ax3a.set_title("(a) S3A Pass 066 Spatial Correlation Matrix $C(s_1, s_2)$\n[Western Bay of Bengal / EICC Regime]")
    fig3.colorbar(im1, ax=ax3a, label="Correlation Coefficient $r$")
    ax3a.axvline(150, color="k", linestyle="--", lw=1.5, label="Coastal Zone (150 km)")
    ax3a.axhline(150, color="k", linestyle="--", lw=1.5)
    ax3a.legend(loc="upper left", fontsize=9)
    
    im2 = ax3b.pcolormesh(
        sla_gfo.x.values,
        sla_gfo.x.values,
        cmat_gfo,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=1.0,
        shading="auto",
    )
    ax3b.set_xlabel("Along-Track Distance $s_1$ (km from Coast)")
    ax3b.set_ylabel("Along-Track Distance $s_2$ (km from Coast)")
    ax3b.set_title("(b) GFO Pass 008 Spatial Correlation Matrix $C(s_1, s_2)$\n[Andaman Sea Basin Mode Regime]")
    fig3.colorbar(im2, ax=ax3b, label="Correlation Coefficient $r$")
    
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "fig3_spatial_correlation_matrix.png", dpi=300)
    plt.close(fig3)
    
    # ---------------------------------------------------------
    # FIGURE 4: Spatial Autocorrelation Functions & Model Fits
    # ---------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Plot S3A ACF
    r_range = np.linspace(0, 1000, 200)
    ax4a.plot(lags_s3a, acf_s3a_tot, "bo-", ms=4, lw=1.5, label="Total SLA (Observed ACF)")
    ax4a.plot(lags_s3a_ns, acf_s3a_ns, "rs--", ms=4, lw=1.5, label="Deseasonalized SLA (Observed ACF)")
    if not np.isnan(scales_s3a_tot["exponential_fit_km"]):
        ax4a.plot(r_range, exp_decay(r_range, scales_s3a_tot["exponential_fit_km"]), "b--", lw=2, label=f"Exp Fit (Total, $L_d$={scales_s3a_tot['exponential_fit_km']:.0f} km)")
    if not np.isnan(scales_s3a_ns["exponential_fit_km"]):
        ax4a.plot(r_range, exp_decay(r_range, scales_s3a_ns["exponential_fit_km"]), "r:", lw=2, label=f"Exp Fit (Deseas, $L_d$={scales_s3a_ns['exponential_fit_km']:.0f} km)")
        
    ax4a.axhline(1.0 / np.e, color="k", linestyle="-.", lw=1.2, label=f"1/e Threshold ({1/np.e:.3f})")
    ax4a.axhline(0.0, color="gray", linestyle=":", lw=1.0)
    ax4a.set_xlim(0, 800)
    ax4a.set_ylim(-0.3, 1.05)
    ax4a.set_xlabel(r"Spatial Separation Distance $\Delta x$ (km)")
    ax4a.set_ylabel(r"Spatial Correlation $R(\Delta x)$")
    ax4a.set_title(f"(a) S3A Pass 066 Spatial ACF & Decorrelation\n[$L_{{1/e}}$={scales_s3a_tot['e_folding_km']:.0f} km (Total), {scales_s3a_ns['e_folding_km']:.0f} km (Deseas)]")
    ax4a.legend(loc="upper right", fontsize=9)
    ax4a.grid(True, linestyle="--", alpha=0.5)
    
    # Plot GFO ACF
    ax4b.plot(lags_gfo, acf_gfo_tot, "bo-", ms=4, lw=1.5, label="Total SLA (Observed ACF)")
    ax4b.plot(lags_gfo_ns, acf_gfo_ns, "rs--", ms=4, lw=1.5, label="Deseasonalized SLA (Observed ACF)")
    if not np.isnan(scales_gfo_tot["exponential_fit_km"]):
        ax4b.plot(r_range, exp_decay(r_range, scales_gfo_tot["exponential_fit_km"]), "b--", lw=2, label=f"Exp Fit (Total, $L_d$={scales_gfo_tot['exponential_fit_km']:.0f} km)")
    if not np.isnan(scales_gfo_ns["exponential_fit_km"]):
        ax4b.plot(r_range, exp_decay(r_range, scales_gfo_ns["exponential_fit_km"]), "r:", lw=2, label=f"Exp Fit (Deseas, $L_d$={scales_gfo_ns['exponential_fit_km']:.0f} km)")
        
    ax4b.axhline(1.0 / np.e, color="k", linestyle="-.", lw=1.2, label=f"1/e Threshold ({1/np.e:.3f})")
    ax4b.axhline(0.0, color="gray", linestyle=":", lw=1.0)
    ax4b.set_xlim(0, 800)
    ax4b.set_ylim(-0.3, 1.05)
    ax4b.set_xlabel(r"Spatial Separation Distance $\Delta x$ (km)")
    ax4b.set_ylabel(r"Spatial Correlation $R(\Delta x)$")
    ax4b.set_title(f"(b) GFO Pass 008 Spatial ACF & Decorrelation\n[Basin-Scale Remote Kelvin Wave Coherence]")
    ax4b.legend(loc="upper right", fontsize=9)
    ax4b.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig4.savefig(OUT_DIR / "fig4_spatial_acf_decorrelation.png", dpi=300)
    plt.close(fig4)
    
    # ---------------------------------------------------------
    # FIGURE 5: Spatiotemporal Lead-Lag Correlation Diagrams
    # ---------------------------------------------------------
    fig5, ((ax5a, ax5b), (ax5c, ax5d)) = plt.subplots(2, 2, figsize=(15, 11))
    
    # S3A Lead-Lag (Total SLA, ref = Near Coast 50km)
    c5a = ax5a.pcolormesh(
        tau_days_s3a,
        sla_s3a.x.values,
        ll_s3a_50km,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.8,
        shading="auto",
    )
    ax5a.axvline(0, color="k", linestyle="--", lw=1.2)
    ax5a.axhline(50, color="k", linestyle=":", lw=1.5, label="Ref Point (50 km)")
    ax5a.set_xlabel("Time Lag $\\tau$ (days) [$\\tau > 0$: Point $x$ Lags Coast]")
    ax5a.set_ylabel("Along-Track Distance (km from Coast)")
    ax5a.set_title("(a) S3A Pass 066 Lead-Lag $R(x, \\tau)$ (Total SLA, Ref: Coast $x=50$ km)")
    fig5.colorbar(c5a, ax=ax5a, label="Correlation $r$")
    ax5a.legend(loc="upper right", fontsize=9)
    
    # S3A Lead-Lag (Deseasonalized SLA)
    c5b = ax5b.pcolormesh(
        tau_days_s3a,
        sla_s3a_ns.x.values,
        ll_s3a_ns_50km,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.8,
        shading="auto",
    )
    ax5b.axvline(0, color="k", linestyle="--", lw=1.2)
    ax5b.axhline(50, color="k", linestyle=":", lw=1.5, label="Ref Point (50 km)")
    ax5b.set_xlabel("Time Lag $\\tau$ (days)")
    ax5b.set_ylabel("Along-Track Distance (km from Coast)")
    ax5b.set_title("(b) S3A Pass 066 Lead-Lag $R(x, \\tau)$ (Deseasonalized SLA)")
    fig5.colorbar(c5b, ax=ax5b, label="Correlation $r$")
    ax5b.legend(loc="upper right", fontsize=9)
    
    # GFO Lead-Lag (Total SLA, ref = Near Coast 50km)
    c5c = ax5c.pcolormesh(
        tau_days_gfo,
        sla_gfo.x.values,
        ll_gfo_50km,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.8,
        shading="auto",
    )
    ax5c.axvline(0, color="k", linestyle="--", lw=1.2)
    ax5c.axhline(50, color="k", linestyle=":", lw=1.5, label="Ref Point (50 km)")
    ax5c.set_xlabel("Time Lag $\\tau$ (days) [$\\tau > 0$: Point $x$ Lags Coast]")
    ax5c.set_ylabel("Along-Track Distance (km from Coast)")
    ax5c.set_title("(c) GFO Pass 008 Lead-Lag $R(x, \\tau)$ (Total SLA, Ref: Coast $x=50$ km)")
    fig5.colorbar(c5c, ax=ax5c, label="Correlation $r$")
    ax5c.legend(loc="upper right", fontsize=9)
    
    # S3A Lead-Lag line profiles at key offshore stations
    sample_dists = [50, 100, 200, 400, 800, 1200]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_dists)))
    for dist_val, col in zip(sample_dists, colors):
        idx_p = np.argmin(np.abs(sla_s3a.x.values - dist_val))
        ax5d.plot(tau_days_s3a, ll_s3a_50km[idx_p, :], "o-", color=col, ms=4, lw=1.5, label=f"$x$ = {sla_s3a.x.values[idx_p]:.0f} km")
    ax5d.axvline(0, color="gray", linestyle="--", lw=1.0)
    ax5d.axhline(0, color="gray", linestyle=":", lw=1.0)
    ax5d.set_xlabel("Time Lag $\\tau$ (days)")
    ax5d.set_ylabel("Correlation with Coast ($x=50$ km)")
    ax5d.set_title("(d) S3A Lead-Lag Correlation Profiles at Offshore Distances")
    ax5d.legend(loc="upper right", fontsize=8.5)
    ax5d.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig5.savefig(OUT_DIR / "fig5_lead_lag_correlations.png", dpi=300)
    plt.close(fig5)
    
    # ---------------------------------------------------------
    # FIGURE 6: Local Decorrelation Scales vs. Coast Distance
    # ---------------------------------------------------------
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Spatial decorrelation length profile
    ax6a.plot(sla_s3a.x.values, local_ld_s3a, "b-", lw=2, label="S3A 066 (Western BoB / EICC)")
    ax6a.plot(sla_gfo.x.values, local_ld_gfo, "m--", lw=2, label="GFO 008 (Andaman Sea)")
    ax6a.axvspan(0, 150, color="orange", alpha=0.15, label="Near-Shore Boundary Zone (0–150 km)")
    ax6a.set_xlabel("Along-Track Distance from Coast (km)")
    ax6a.set_ylabel("Local Spatial Decorrelation Length $L_d(x)$ (km)")
    ax6a.set_title("(a) Along-Track Spatial Decorrelation Length Profile")
    ax6a.set_ylim(0, 600)
    ax6a.legend(loc="upper right")
    ax6a.grid(True, linestyle="--", alpha=0.5)
    
    # Temporal Autocorrelation (Average) Profile
    mean_temp_s3a = np.nanmean(temp_acf_s3a, axis=0)
    mean_temp_gfo = np.nanmean(temp_acf_gfo, axis=0)
    ax6b.plot(t_days_s3a, mean_temp_s3a, "bo-", ms=5, lw=1.8, label="S3A 066 Mean Temporal ACF")
    ax6b.plot(t_days_gfo, mean_temp_gfo, "ms--", ms=5, lw=1.8, label="GFO 008 Mean Temporal ACF")
    ax6b.axhline(1.0 / np.e, color="k", linestyle="-.", lw=1.2, label=f"1/e Threshold ({1/np.e:.3f})")
    ax6b.axhline(0.0, color="gray", linestyle=":", lw=1.0)
    ax6b.set_xlabel("Time Lag $\\tau$ (days)")
    ax6b.set_ylabel("Temporal Autocorrelation $R_t(\\tau)$")
    ax6b.set_title("(b) Temporal Persistence & Decorrelation Time Scale")
    ax6b.legend(loc="upper right")
    ax6b.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig6.savefig(OUT_DIR / "fig6_local_decorrelation_scales.png", dpi=300)
    plt.close(fig6)
    
    print("\nAll figures and tables generated successfully!")
    print(f"Results saved in: {OUT_DIR}")
    return df_metrics


if __name__ == "__main__":
    run_full_analysis()
