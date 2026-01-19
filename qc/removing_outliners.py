import matplotlib.pyplot as plt    
import xarray as xr    
import numpy as np    
import sys

def clean_mean(sample,cutoff):    
    mn=np.mean(sample)    
    print(mn)    
    std=np.std(sample)    
    print(std)    
    clean_sample=[]    
    for i in sample:    
        if i > mn-cutoff*std and i<mn+cutoff*std:    
            clean_sample.append(i)    
    clean_mean=np.mean(clean_sample)    
    return clean_mean    
    
sample=[1,2,3,4,5,6,7,8,9,10,100]    
cutoff=3    
out=clean_mean(sample,cutoff)    
print(out)  

ds=xr.open_dataset("2902093_Sprof.nc")    
print(ds)    
# for v in ds.variabes:    
    # print(v)    
print(ds.JULD)    
ds=ds.swap_dims({"N_PROF":"JULD"})    
ds=ds.rename({"JULD":"time"})    
ds=ds.rename({"N_LEVELS":"depth"})    
# ds.TEMP.sel(depth=slice(0,100)).plot(x="time",yincrease=False)    
arr=ds.TEMP.sel(depth=slice(0,10)).to_numpy()
arr=ds.TEMP.sel(depth=100).to_numpy()
# arr=arr1.flatten()
print(arr)
print(np.nanmean(arr))
print(np.nanmax(arr))
print(np.nanmin(arr))
r1=(np.nanmin(arr))
r2=(np.nanmax(arr))
hist, bin_edges = np.histogram(arr,50,(r1,r2))
print(hist.shape)
print(bin_edges.shape)
plt.bar(bin_edges[:-1],hist)
plt.axvline(x = np.nanmean(arr), color = 'b')
plt.show()
sys.exit()
