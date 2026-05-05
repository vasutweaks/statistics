import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

import geocat.viz as gv


def main():
    # Reference standard deviation and one test point (sigma_f, R)
    sigma_r = 1.0
    sigma_f = 1.25
    corr = 0.75

    theta = np.arccos(corr)
    x_test = sigma_f * corr
    y_test = sigma_f * np.sin(theta)

    fig = plt.figure(figsize=(8, 8))
    taylor = gv.TaylorDiagram(fig=fig, label="REF")
    ax = taylor.ax

    # Empty Taylor diagram framework: correlation grid + RMS contours
    taylor.add_corr_grid(np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95]))
    taylor.add_contours(levels=np.arange(0.2, 2.1, 0.2), colors="lightgray", linewidths=0.8)

    # Mark reference and test points
    ax.plot(sigma_r, 0.0, marker="o", color="black", markersize=7)
    ax.text(sigma_r + 0.03, 0.02, "Ref", fontsize=11)
    ax.plot(x_test, y_test, marker="o", color="crimson", markersize=8)
    ax.text(x_test + 0.03, y_test + 0.03, "Test", color="crimson", fontsize=11)

    # sigma_f: line from origin to test point
    ax.plot([0.0, x_test], [0.0, y_test], color="crimson", linewidth=2)
    ax.text(0.55 * x_test, 0.55 * y_test + 0.03, r"$\sigma_f$", color="crimson", fontsize=12)

    # sigma_r: line from origin to reference point
    ax.plot([0.0, sigma_r], [0.0, 0.0], color="black", linewidth=2)
    ax.text(0.5 * sigma_r, -0.06, r"$\sigma_r$", ha="center", fontsize=12)

    # E': centered RMS difference (line between ref and test)
    ax.plot([sigma_r, x_test], [0.0, y_test], color="royalblue", linewidth=2)
    ax.text((sigma_r + x_test) / 2 + 0.03, y_test / 2 + 0.03, r"$E'$", color="royalblue", fontsize=12)

    # Angle at origin: cos^{-1}(R)
    angle_deg = np.degrees(theta)
    arc = Arc((0.0, 0.0), width=0.5, height=0.5, theta1=0.0, theta2=angle_deg, color="darkgreen", linewidth=1.8)
    ax.add_patch(arc)
    ax.text(0.32 * np.cos(theta / 2), 0.32 * np.sin(theta / 2), r"$\cos^{-1}(R)$", color="darkgreen", fontsize=11)

    ax.set_title("Schematic Taylor Diagram (GeoCAT-viz)", fontsize=13)
    plt.show()


if __name__ == "__main__":
    main()
