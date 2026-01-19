import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt

# ----------------------------------------------------
# 1. Read CSV
# ----------------------------------------------------
file_path = "weight-height.csv"
df = pd.read_csv(file_path)

# ----------------------------------------------------
# 2. Select men's heights
# ----------------------------------------------------
men_heights = df.loc[df["Gender"] == "Male", "Height"].values

# ----------------------------------------------------
# 3. Compute mean and standard deviation
# ----------------------------------------------------
mu = np.mean(men_heights)
sigma = np.std(men_heights, ddof=0)  # population std

print(f"Mean height (mu)      : {mu:.3f}")
print(f"Std deviation (sigma) : {sigma:.3f}")

# ----------------------------------------------------
# 4. Histogram of men's heights
# ----------------------------------------------------
plt.figure()
count, bins, _ = plt.hist(
    men_heights,
    bins=40,
    density=True,
    alpha=0.6
)

# ----------------------------------------------------
# 5. Normal distribution with same mean & std
# ----------------------------------------------------
x = np.linspace(bins[0], bins[-1], 500)
normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((x - mu) / sigma) ** 2)

plt.plot(x, normal_pdf, linewidth=2)

plt.xlabel("Height")
plt.ylabel("Probability density")
plt.title("Men's Heights with Normal Distribution Overlay")
plt.show()

# ----------------------------------------------------
# 6. Percentage of observed values outside ±3.5σ
# ----------------------------------------------------
z_scores = (men_heights - mu) / sigma
outside_data = np.sum(np.abs(z_scores) > 3.5)

pct_outside_data = 100 * outside_data / len(men_heights)

print(f"Observed % outside ±3.5σ : {pct_outside_data:.6f}%")

# ----------------------------------------------------
# 7. Theoretical percentage outside ±3.5σ (Normal)
# ----------------------------------------------------
# CDF of standard normal using error function
def normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))

theoretical_outside = 2 * (1 - normal_cdf(3.5))
pct_outside_theoretical = 100 * theoretical_outside

print(f"Theoretical % outside ±3.5σ (Normal) : {pct_outside_theoretical:.6f}%")

