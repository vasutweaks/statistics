import numpy as np
import matplotlib.pyplot as plt


def tricube_weighting_function(x, bandwidth):
    u = np.abs(x) / bandwidth
    return np.where(u <= 1, (1 - u**3) ** 3, 0)


# def tricube_weighting_function(xd):
#     return (1 - abs(xd)**3)**3
#
# Generate data for x-axis
x_values = np.linspace(-5, 5, 1000)

# Set bandwidth parameter
bandwidth = 1.0

# Calculate tri-cube weighting function values
weights = tricube_weighting_function(x_values, bandwidth)
# weights = tricube_weighting_function(x_values)

# Plotting the tri-cube weighting function
plt.plot(x_values, weights, label="Tri-Cube Weighting Function")
plt.title("Tri-Cube Weighting Function")
plt.xlabel("x")
plt.ylabel("Weight")
plt.grid(True)
plt.legend()
plt.show()
