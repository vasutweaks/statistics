import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find
import pycwt

# copilot --resume=c5161e73-97f4-4a7d-836b-d620eda107a1
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ds = xr.open_dataset('/home/srinivasu/allData/rama/sst/sst0n67e_dy.cdf')

# Squeeze out depth/lat/lon singleton dims → 1-D time series
sst = ds['T_25'].squeeze(drop=True)
sst = sst.sel(time=slice(None, '2020-12-31'))  # 20 years of data
wavelet = pycwt.Morlet(6)             # Instantiate mother wavelet.
result = wavelet.run(sst)              # Run wavelet analysis.
result.plot()
