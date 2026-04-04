"""Plot RAMA SST time series and Lomb periodogram with annotated periodicities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import find_peaks, lombscargle

INPUT_FILE = Path("/home/srinivasu/allData/rama/sst/sst0n90e_dy.cdf")
SST_VARIABLE = "T_25"  # Set to None to auto-pick the first variable matching T_*
OUTPUT_FILE = Path("/home/srinivasu/statistics/periodogram/rama_sst_lomb.png")

# If True, annotate strongest Lomb peaks automatically.
ANNOTATE_AUTO_PEAKS = True
N_AUTO_PEAKS = 5

# Add your custom periods here (in days). These are always annotated.
# Example: [7.0, 14.0, 30.0, 90.0, 365.0]
CUSTOM_PERIODICITIES_DAYS = [30.0, 60.0, 90.0, 182.0, 365.0]
# CUSTOM_PERIODICITIES_DAYS = []


def to_days_since_start(time_values: np.ndarray) -> np.ndarray:
    """Convert datetime64 array to days since first sample."""
    t0 = time_values[0]
    return (time_values - t0) / np.timedelta64(1, "D")


def prepare_series(ds: xr.Dataset, var_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return clean (time_days, data) arrays with finite SST values."""
    da = ds[var_name].squeeze(drop=True)
    if "time" not in da.dims:
        raise ValueError(f"Variable {var_name!r} has no 'time' dimension.")

    time = ds["time"].values
    values = da.values.astype(float)

    # Remove typical missing and impossible SST values.
    mask = np.isfinite(values)
    mask &= values < 1e30
    mask &= (values > -5.0) & (values < 45.0)

    time = time[mask]
    values = values[mask]

    if values.size < 10:
        raise ValueError("Too few valid data points after filtering.")

    return to_days_since_start(time), values


def compute_lomb(t_days: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lomb-Scargle power over a physically useful period range."""
    y_detrended = y - np.nanmean(y)

    dt = np.diff(np.sort(t_days))
    median_dt = np.median(dt[dt > 0])
    nyquist_cpd = 0.5 / median_dt

    # Search ~2 days to ~8 years (or shorter if record is short).
    max_period_days = min(3000.0, 0.95 * (t_days.max() - t_days.min()))
    min_period_days = 2.0

    min_freq_cpd = 1.0 / max_period_days
    max_freq_cpd = min(nyquist_cpd * 0.95, 1.0 / min_period_days)

    freqs_cpd = np.linspace(min_freq_cpd, max_freq_cpd, 8000)
    ang_freq = 2.0 * np.pi * freqs_cpd

    power = lombscargle(t_days, y_detrended, ang_freq, normalize=True)
    periods_days = 1.0 / freqs_cpd

    return periods_days, power


def dominant_periods(periods_days: np.ndarray, power: np.ndarray, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Pick top-N separated peaks from Lomb spectrum."""
    order = np.argsort(periods_days)
    p_sorted = periods_days[order]
    pow_sorted = power[order]

    peaks, _ = find_peaks(pow_sorted, distance=max(5, len(pow_sorted) // 80))
    if peaks.size == 0:
        idx = np.argsort(pow_sorted)[-n:]
        return p_sorted[idx], pow_sorted[idx]

    peak_p = p_sorted[peaks]
    peak_pow = pow_sorted[peaks]

    top = np.argsort(peak_pow)[-n:]
    return peak_p[top], peak_pow[top]


def plot_figure(
    ds: xr.Dataset,
    var_name: str,
    periods_days: np.ndarray,
    power: np.ndarray,
    marked_periods: np.ndarray,
    output: Path,
) -> None:
    """Create 2-panel plot with annotated periodicities."""
    da = ds[var_name].squeeze(drop=True)
    time = ds["time"].values
    y = da.values.astype(float)

    valid = np.isfinite(y) & (y < 1e30) & (y > -5.0) & (y < 45.0)
    t_valid = time[valid]
    y_valid = y[valid]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, constrained_layout=True)

    axes[0].plot(t_valid, y_valid, color="tab:blue", lw=1.1)
    axes[0].set_title(f"RAMA SST Time Series ({var_name})")
    axes[0].set_ylabel("Temperature (degC)")
    axes[0].grid(True, alpha=0.3)

    idx = np.argsort(periods_days)
    p_sorted = periods_days[idx]
    pow_sorted = power[idx]

    axes[1].plot(p_sorted, pow_sorted, color="tab:red", lw=1.1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Period (days, log scale)")
    axes[1].set_ylabel("Normalized Lomb Power")
    axes[1].set_title("Lomb-Scargle Periodogram")
    axes[1].grid(True, which="both", alpha=0.3)

    for p in sorted(marked_periods):
        axes[1].axvline(p, color="k", ls="--", lw=0.9, alpha=0.75)
        axes[1].text(
            p,
            axes[1].get_ylim()[1] * 0.9,
            f"{p:.1f} d",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="k",
        )

    fig.suptitle("RAMA 0N,67E SST: Time Series and Dominant Periodicities", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.show()
    plt.close(fig)


def choose_sst_var(ds: xr.Dataset, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in ds.variables:
            raise ValueError(f"Requested variable {explicit!r} not present in dataset.")
        return explicit

    for name in ds.data_vars:
        if name.upper().startswith("T_"):
            return name

    raise ValueError("Could not infer SST variable. Pass --var explicitly.")


def main() -> None:
    ds = xr.open_dataset(INPUT_FILE)
    var_name = choose_sst_var(ds, SST_VARIABLE)

    t_days, y = prepare_series(ds, var_name)
    periods_days, power = compute_lomb(t_days, y)
    marked_periods = np.array([], dtype=float)

    if ANNOTATE_AUTO_PEAKS:
        auto_periods, _ = dominant_periods(periods_days, power, n=N_AUTO_PEAKS)
        marked_periods = np.concatenate([marked_periods, auto_periods])

    if len(CUSTOM_PERIODICITIES_DAYS) > 0:
        custom = np.array(CUSTOM_PERIODICITIES_DAYS, dtype=float)
        custom = custom[np.isfinite(custom) & (custom > 0)]
        marked_periods = np.concatenate([marked_periods, custom])

    # Unique and keep only periods inside computed spectrum range.
    if marked_periods.size > 0:
        pmin, pmax = periods_days.min(), periods_days.max()
        marked_periods = np.unique(marked_periods)
        marked_periods = marked_periods[(marked_periods >= pmin) & (marked_periods <= pmax)]

    plot_figure(ds, var_name, periods_days, power, marked_periods, OUTPUT_FILE)

    print(f"Saved figure: {OUTPUT_FILE}")
    print("Annotated periodicities (days):", ", ".join(f"{p:.2f}" for p in sorted(marked_periods)))


if __name__ == "__main__":
    main()
