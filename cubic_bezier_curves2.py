import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

# Generate a set of random points
np.random.seed(43)  # For reproducibility
num_points = 100
x = np.sort(np.random.rand(num_points))
y = np.random.rand(num_points)

# Define the number of knots and degree of the spline
k = 3  # Degree of the B-spline
t = np.linspace(0, 1, num_points - k + 1)  # Knot vector
t = np.concatenate(([0] * k, t, [1] * k))  # Extend the knot vector

# Fit B-spline to the points
spl = make_lsq_spline(x, y, t, k)

# Generate points on the B-spline
x_spline = np.linspace(0, 1, 100)
y_spline = spl(x_spline)

# Plot the original points and the B-spline
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'ro', label='Data points')
plt.plot(x_spline, y_spline, 'b-', label='B-spline fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('B-spline fitting to random points')
plt.grid(True)
plt.show()
