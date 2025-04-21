import numpy as np
import matplotlib.pyplot as plt
import sys


# # Load the .npy file
beeta_values = np.load('final_results/20250306_184702_beta_values.npy')
gamma_values = np.load('final_results/20250306_184702_gamma_values.npy')
phi_values = np.load('final_results/20250306_184702_phi_values.npy')
min_values = np.load('final_results/20250306_184702_min_dist_from_beta_values.npy')


# Print the contents
print(beeta_values)
print(beeta_values.shape)

print(gamma_values)
print(gamma_values.shape)

print(phi_values)
print(phi_values.shape)

# Define time factor
iterations = np.arange(len(beeta_values))
time_factor = 5  # seconds
time_values = iterations * time_factor


# Plot Min Distance
plt.figure(figsize=(8, 5))
# plt.plot(time_values, beeta_values, label='β', marker='o', linestyle='-')
plt.plot(time_values, min_values, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Min Distance (m)")
# plt.title("Beeta")
# plt.legend()
plt.grid(True)
plt.xlim(0, max(time_values))

# Plot Beeta
plt.figure(figsize=(8, 5))
# plt.plot(time_values, beeta_values, label='β', marker='o', linestyle='-')
plt.plot(time_values, beeta_values, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("β")
# plt.title("Beeta")
# plt.legend()
plt.grid(True)
plt.xlim(0, max(time_values))
plt.savefig('graph/beta_plot.pdf', format='pdf', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.savefig('graph/beta_plot.png', format='png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.show()



# Create figure with primary y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Min Distance on primary y-axis (left)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Min Distance (m)')
ax1.plot(time_values, min_values,  marker='o', linestyle='-', label='Min Distance')
ax1.set_ylim(3, 15)
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)

# Create secondary y-axis (right) for Beta
ax2 = ax1.twinx()
ax2.set_ylabel('β')
ax2.plot(time_values, beeta_values, color='red', marker='o', linestyle='-', label='β')
ax2.tick_params(axis='y')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Set x-axis limits
ax1.set_xlim(0, max(time_values))

# Add title
# plt.title('Min Distance and Beta vs. Time')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('graph/min_dist_and_beta_sim.pdf', format='pdf', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.savefig('graph/min_dist_and_beta_sim.png', format='png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)

plt.show()


# Plot Gamma
plt.figure(figsize=(8, 5))
# plt.plot(time_values, gamma_values, label='γ', marker='o', linestyle='-')
plt.plot(time_values, gamma_values, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("γ")
# plt.title("Gamma")
# plt.legend()
plt.grid(True)
plt.xlim(0, max(time_values))
plt.savefig('graph/gamma_plot_sim.pdf', format='pdf', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.savefig('graph/gamma_plot_sim.png', format='png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot Phi
plt.figure(figsize=(8, 5))
# plt.plot(time_values, phi_values, label='φ', marker='o', linestyle='-')
plt.plot(time_values, phi_values, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("φ")
# plt.title("Phi")
# plt.legend()
plt.grid(True)
plt.xlim(0, max(time_values))
plt.savefig('graph/phi_plot_sim.pdf', format='pdf', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.savefig('graph/phi_plot_sim.png', format='png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.show()
