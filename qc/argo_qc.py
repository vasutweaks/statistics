import matplotlib.pyplot as plt    
import xarray as xr    
import numpy as np    
import sys

ds=xr.open_dataset("2902093_Sprof.nc")    
print(ds)    
# for v in ds.variabes:    
    # print(v)    
print(ds.JULD)    
ds=ds.swap_dims({"N_PROF":"JULD"})    
ds=ds.rename({"JULD":"time"})    
ds=ds.rename({"N_LEVELS":"depth"})    
# ds.TEMP.sel(depth=slice(0,100)).plot(x="time",yincrease=False)    
# arr=ds.TEMP.sel(depth=slice(0,10)).to_numpy()
# fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2,figsize=(12,12))
fig, axs1=plt.subplots(4,4,figsize=(12,12))
axs=axs1.flat
# arr=arr1.flatten()
# print(arr)
# print(np.nanmean(arr))
# print(np.nanmax(arr))
# print(np.nanmin(arr))

dpths=np.linspace(5,165,17)
print(dpths)

i=0
for d in dpths[:-1]:
    arr=ds.TEMP.sel(depth=int(d)).to_numpy()
    r1=(np.nanmin(arr))
    r2=(np.nanmax(arr))
    hist, bin_edges = np.histogram(arr,50,(r1,r2))
    print(hist.shape)
    print(bin_edges.shape)
    axs[i].bar(bin_edges[:-1],hist)
    axs[i].axvline(x = np.nanmean(arr), color = 'b')
    d1=str(d)
    axs[i].set_title(d1)
    i=i+1
plt.show()
