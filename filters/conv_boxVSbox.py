import numpy as np
import matplotlib.pyplot as plt


# Function to create a box signal
def box_signal(length, width):
    signal = np.zeros(length)
    signal[:width] = 1
    return signal


# Length of the signals
length = 21

# Width of the box signals
width = 5

# Create the first box signal
box1 = box_signal(length, width)

# Create the second box signal by flipping the first one
box2 = np.flip(box1)

# Convolve the two box signals
convolution_result = np.convolve(box1, box2, mode="same")

# Plot the signals and the convolution result
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(np.arange(length), box1)
plt.title("Box Signal 1")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.stem(np.arange(length), box2)
plt.title("Box Signal 2")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.stem(np.arange(length), convolution_result)
plt.title("Convolution Result")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
