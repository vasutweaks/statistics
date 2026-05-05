import matplotlib.pyplot as plt
import numpy as np
import geocat.viz as gv

fig = plt.figure(figsize=(10, 10))
dia = gv.TaylorDiagram(fig=fig, label='REF')

# Reference point is at r=1, theta=0
# Test point at r=1.2, theta=arccos(0.8)
r_r = 1.0
theta_r = 0

r_f = 1.2
R = 0.8
theta_f = np.arccos(R)

print(f"Theta_f: {theta_f}")

# Try plotting on dia.ax
# We need to know if dia.ax expects (theta, r)
dia.ax.plot([theta_r, theta_f], [r_r, r_f], 'ro-')
dia.ax.plot([0, theta_f], [0, r_f], 'g-')
dia.ax.plot([0, theta_r], [0, r_r], 'b-')

plt.show()
plt.savefig('test_taylor.png')
