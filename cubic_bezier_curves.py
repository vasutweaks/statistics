import matplotlib.pyplot as plt
import numpy as np

# Define the control points
P0 = np.array([1, 1])
P1 = np.array([2, 3])
P2 = np.array([3, 3])
P3 = np.array([4, 1])


# Define the Bézier curve function
def bezier_curve_point(t, P0, P1, P2, P3):
    return (
        (1 - t) ** 3 * P0
        + 3 * (1 - t) ** 2 * t * P1
        + 3 * (1 - t) * t**2 * P2
        + t**3 * P3
    )


# Generate points on the curve
t_values = np.linspace(0, 1, 100)
curve_points = np.array([bezier_curve_point(t, P0, P1, P2, P3) for t in t_values])
print(curve_points)

# Plot the curve
plt.plot(curve_points[:, 0], curve_points[:, 1], label="Cubic Bézier Curve")

# Plot the control points
control_points = np.array([P0, P1, P2, P3])
plt.plot(
    control_points[:, 0], control_points[:, 1], "ro-", label="Control Points"
)

# Add labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Cubic Bézier Curve")
plt.grid(True)
plt.axis("equal")

# Show the plot
plt.show()
