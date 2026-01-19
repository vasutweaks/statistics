import numpy as np
import matplotlib.pyplot as plt


# Define a sample function
def sample_function(n):
    return np.sin(n) + np.cos(n)


# Generate sample input signal
x = np.arange(-10, 11)  # Sample range
f_x = sample_function(x)

# Define the unit impulse function
delta = np.zeros_like(x)
delta[len(x) // 2] = 1  # Setting the impulse at the center of the array

# Perform convolution
conv_result = np.convolve(f_x, delta, mode="same")

# Plot original function and convolution result
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.stem(x, f_x)
plt.title("Original Function")
plt.xlabel("n")
plt.ylabel("f[n]")

plt.subplot(2, 1, 2)
plt.stem(x, conv_result)
plt.title("Convolution with Unit Impulse")
plt.xlabel("n")
plt.ylabel("(f * delta)[n]")

plt.tight_layout()
plt.show()
