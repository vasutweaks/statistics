import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# def gaussian_filter(arr, sigma):
#     # Create a 1D Gaussian kernel
#     size = int(6 * sigma + 1)
#     kernel = np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2))
#     kernel /= np.sum(kernel)  # Normalize the kernel
#     # Convolve the array with the Gaussian kernel
#     filtered_arr = np.convolve(arr, kernel, mode='same')
#     return filtered_arr
#
# Generating a random 1D array
np.random.seed(0)
arr = np.random.randn(100)

# Plotting the original and smoothed arrays
plt.figure(figsize=(10, 6))
plt.plot(arr, label="Original")
# Applying Gaussian filter
sigma = 1  # Standard deviation of the Gaussian kernel
arr_smoothed = gaussian_filter(arr, sigma)
plt.plot(arr_smoothed, label=f"Smoothed (σ={sigma})")
sigma = 2  # Standard deviation of the Gaussian kernel
arr_smoothed = gaussian_filter(arr, sigma)
plt.plot(arr_smoothed, label=f"Smoothed (σ={sigma})")
sigma = 3  # Standard deviation of the Gaussian kernel
arr_smoothed = gaussian_filter(arr, sigma)
plt.plot(arr_smoothed, label=f"Smoothed (σ={sigma})")
plt.legend()
plt.title("1D Gaussian Filter Example")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
