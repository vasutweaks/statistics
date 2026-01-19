import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Example data
x = np.linspace(0, 10, 10)
y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)

# Generate a cubic B-spline representation of an N-D curve
tck, u = splprep([x, y], s=2)

# Evaluate the B-spline or its derivatives
x_smooth, y_smooth = splev(np.linspace(0, 1, 100), tck)

# Plot the original data
plt.scatter(x, y, label='Original Data')

# Plot the smoothed data
plt.plot(x_smooth, y_smooth, 'r-', label='Smoothed Data')

# Adding labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Smoothing using Bezier Curves')
plt.legend()
plt.show()

