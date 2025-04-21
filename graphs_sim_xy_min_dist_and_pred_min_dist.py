import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist

# Define the dataset
data1 = np.load('data/20250306_184702_min_dist_values.npy')
data2 = np.load('data/20250306_184702_min_dist_from_beta_values.npy')

print(data1.shape)
print(data2.shape)

# Create final plots
plt.figure(figsize=(10, 5))
plt.plot(data1, marker="o", linestyle="-")
plt.plot(data2, marker="o", linestyle="-")
plt.xlabel("Time (s)")
plt.ylabel("Min Distance (m)")
plt.grid(True)
# plt.xlim(0, max(time_values))
# plt.ylim(0, max(df["Min Distance"]) * 1.1)

# plt.savefig('graph/minimum_distance_plot_sim.pdf', format='pdf', dpi=300, 
#             bbox_inches='tight', pad_inches=0.1)
# plt.savefig('graph/minimum_distance_plot_sim.png', format='png', dpi=300, 
#             bbox_inches='tight', pad_inches=0.1)
plt.show()

