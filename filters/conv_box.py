import numpy as np
import matplotlib.pyplot as plt

# Define a sample signal
np.random.seed(0)
arr = np.random.rand(100)  # Example 1D numpy array

# Define the window size for moving average
window_size = 12

# Generate the box signal of width 12
box_signal = np.ones(window_size) / window_size

# Calculate moving average using np.convolve
moving_average_result = np.convolve(arr, box_signal, mode="valid")

# Plot original signal and moving average result
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(arr, label="Original Signal")
plt.title("Original Signal")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(moving_average_result, label="Moving Average (Convolution)")
plt.title("Moving Average (Convolution)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()
