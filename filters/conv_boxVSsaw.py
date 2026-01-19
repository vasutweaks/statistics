import numpy as np
import matplotlib.pyplot as plt

# Define the length of the signals
n = 1000

# Create the box signal
box_signal = np.zeros(n)
box_signal[400:600] = 1  # Box signal with value 1 between indices 30 and 70

# Create the saw signal
saw_signal = np.zeros(n)
saw_signal[400:600] = np.linspace(0, 1, 200)

# Perform convolution
convolved_signal = np.convolve(box_signal, saw_signal, mode='same')/len(box_signal)

# Plot the signals and the convolved signal
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(box_signal)
plt.title('Box Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(saw_signal)
plt.title('Saw Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(convolved_signal)
plt.title('Convolved Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
