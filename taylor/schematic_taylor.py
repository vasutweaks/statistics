import matplotlib.pyplot as plt
import numpy as np
import geocat.viz as gv

# 1. Create figure and TaylorDiagram instance
fig = plt.figure(figsize=(8, 8))
# Setting refstd=1.0 for a normalized Taylor diagram
dia = gv.TaylorDiagram(fig=fig, label='REF', refstd=1.0)

# Add correlation grid
dia.add_corr_grid(np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]))

# Add standard deviation grid
dia.add_std_grid(np.array([0.5, 1.0, 1.5]))

# Add RMSE contours (Centered RMS difference)
dia.add_contours(levels=np.arange(0, 1.6, 0.25), colors='lightgrey', linewidths=0.5)

# 2. Take a test point on the taylor diagram
# Let's say standard deviation sigma_f = 1.2 and correlation R = 0.8
sigma_f = 1.2
R = 0.8
theta_f = np.arccos(R)

# Reference point standard deviation
sigma_r = 1.0
theta_r = 0.0

# Add the test point
dia.add_model_set([sigma_f], [R], color='red', marker='o', s=100, label='Test Point', annotate_on=False)

# 3. Join the test point and origin and label the line sigma f
# In TaylorDiagram, dia.ax is the axes where we plot (theta, radius)
dia.ax.plot([0, theta_f], [0, sigma_f], color='red', linestyle='--', linewidth=2)
# Label sigma_f slightly offset from the line
dia.ax.text(theta_f + 0.1, sigma_f / 2, r'$\sigma_f$', fontsize=20, color='red', verticalalignment='center')

# 4. Join the reference point on the x-axis with origin labeling it sigma r
dia.ax.plot([0, theta_r], [0, sigma_r], color='blue', linestyle='--', linewidth=2)
# Label sigma_r below the x-axis
dia.ax.text(theta_r + 0.15, sigma_r / 2, r'$\sigma_r$', fontsize=20, color='blue', verticalalignment='top', horizontalalignment='right')

# 5. The line joining the ref point and test point is labelled as E prime
dia.ax.plot([theta_r, theta_f], [sigma_r, sigma_f], color='green', linestyle='-', linewidth=2)
# Label E' near the midpoint of the line joining REF and MOD
mid_theta_E = (theta_r + theta_f) / 2
mid_sigma_E = (sigma_r + sigma_f) / 2
dia.ax.text(mid_theta_E + 0.1, mid_sigma_E + 0.05, r"$E'$", fontsize=20, color='green')

# 6. The angle at the origin is labelled as cos inverse R
# Draw an arc for the angle
arc_theta = np.linspace(0, theta_f, 50)
arc_r = np.full_like(arc_theta, 0.1)
dia.ax.plot(arc_theta, arc_r, color='black', linewidth=1)
# Label the angle
dia.ax.text(theta_f / 4, 0.20, r'$\cos^{-1} R$', fontsize=16, horizontalalignment='center', verticalalignment='bottom')

# Add a title
plt.title("Schematic Taylor Diagram", size=20, pad=30)

# Show the plot
plt.tight_layout()
plt.show()
