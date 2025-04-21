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
data = np.load('final_results/20250320_082540_x_array.npy')
time_val = np.load('final_results/20250320_082540_timing_values.npy')


print(data.shape)
print(data)

print("---"*10)

print(time_val.shape)
print(time_val)

# Extract x and y coordinates
x_coords_ = data[0, :5, :]  # First row → x-coordinates (4 robots and 1 obstacle)
y_coords_ = data[1, :5, :]  # Second row → y-coordinates (4 robots and 1 obstacle)


# Stack along rows (axis 0)
position = np.vstack((x_coords_, y_coords_)) 

# print(position.shape)
# print(position)


def find_nearest_point_all_cal(ellipse_params, point_of_interested, searching_angle):
    # Define parameters
    center_x, center_y, a, b, phi = ellipse_params
    x0, y0 = point_of_interested
    
    # Define the parametric equations for x(t) and y(t)
    safety_factor = 1  # This prevents the nearest points located on the boundary of the ellipse
    a = safety_factor * a
    b = safety_factor * b
    
    def x(t):
        return center_x + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    
    def y(t):
        return center_y + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    
    # Define the function to calculate the distance between two points
    def distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Initialize minimum distance and t
    min_distance = a
    t_min = 0
    
    # Find minimum distance
    for t in searching_angle:
        dist = distance(x(t), y(t), x0, y0)
        if dist < min_distance:
            min_distance = dist
            t_min = t
    
    # The nearest point
    nearest_point = [x(t_min), y(t_min)]
    return nearest_point, min_distance

def cal_nearest_point_to_boundary(point_x, point_y, center_x, center_y, a, b, phi):
    # Define the search interval
    interval = 1e-2
    searching_angle = np.arange(0, 2*np.pi, interval)
    ellipse_params = [center_x, center_y, a, b, phi]
    
    nearest_point, min_distance = find_nearest_point_all_cal(ellipse_params, [point_x, point_y], searching_angle)
    nearest_point = np.array(nearest_point)
    nearest_point = np.reshape(nearest_point, (2,1))
    
    return nearest_point, min_distance

# Create 'graph' directory if it doesn't exist
if not os.path.exists('graph'):
    os.makedirs('graph')

# Define ellipse parameters
origin_x_ellipse = -95
origin_y_ellipse = -75
major_axis_ellipse = 50  # semi-major axis 70
minor_axis_ellipse = 30  # semi-minor axis 50
rotational_angle_ellipse = np.radians(45)  # rotation angle in degrees

# Define time factor
time_factor = 1  # seconds

# Generate ellipse points
theta = np.linspace(0, 2 * np.pi, 300)
x_ellipse = major_axis_ellipse * np.cos(theta)
y_ellipse = minor_axis_ellipse * np.sin(theta)

# Rotation matrix
R = np.array([[np.cos(rotational_angle_ellipse), -np.sin(rotational_angle_ellipse)],
              [np.sin(rotational_angle_ellipse), np.cos(rotational_angle_ellipse)]])
rotated_ellipse = R @ np.array([x_ellipse, y_ellipse])

# Translate to the origin
x_ellipse_final = rotated_ellipse[0, :] + origin_x_ellipse
y_ellipse_final = rotated_ellipse[1, :] + origin_y_ellipse



# # Extract x and y coordinates
x_coords = position[:5, :]
y_coords = position[5:10, :]

# Number of time steps
num_frames = x_coords.shape[1]
print("num_frames: ", num_frames)

# Define colors for robots
colors = ['b', 'g', '#FFA500', 'purple', 'red']

class AnimationManager:
    def __init__(self, fig, ax, x_coords, y_coords, x_ellipse_final, y_ellipse_final, 
                 origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, 
                 rotational_angle_ellipse):
        self.fig = fig
        self.ax = ax
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.num_frames = x_coords.shape[1]
        self.min_distance_values = []
        self.processed_frames = set()
        
        # Store ellipse parameters
        self.x_ellipse_final = x_ellipse_final
        self.y_ellipse_final = y_ellipse_final
        self.origin_x_ellipse = origin_x_ellipse
        self.origin_y_ellipse = origin_y_ellipse
        self.major_axis_ellipse = major_axis_ellipse
        self.minor_axis_ellipse = minor_axis_ellipse
        self.rotational_angle_ellipse = rotational_angle_ellipse
        
        # Initialize plots
        self.colors = ['b', 'g', '#FFA500', 'purple', 'red']
        self.lines = [ax.plot([], [], marker='o', linestyle='-', color=self.colors[i], 
                     label=f'Robot {i+1}')[0] for i in range(4)]
        self.obstacle_marker = ax.scatter([], [], color='red', marker='H', s=100, 
                                        label='Obstacle')
        self.start_markers = [ax.scatter([], [], color=self.colors[i], marker='s', s=80) 
                            for i in range(5)]
        self.end_markers = [ax.scatter([], [], color=self.colors[i], marker='*', s=150) 
                          for i in range(5)]
        self.min_dist_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                                   fontsize=12, verticalalignment='top')
        self.min_dist_line, = ax.plot([], [], 'k--', linewidth=2)
        self.boundary_crosses = [ax.scatter([], [], color=self.colors[i], marker='x', 
                               s=100, label=f'CP-R{i+1}') for i in range(4)]

    def update(self, frame):
        print(f"Processing frame {frame}")
        
        if frame >= self.num_frames:
            return self.get_artists()
            
        if frame in self.processed_frames:
            print(f"Frame {frame} already processed, skipping")
            return self.get_artists()
            
        self.processed_frames.add(frame)
        
        # Get current positions
        positions = np.column_stack((self.x_coords[:, frame], self.y_coords[:, frame]))
        
        # Calculate robot-robot distances
        robot_robot_distances = []
        robot_pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                dist = np.linalg.norm(positions[i] - positions[j])
                robot_robot_distances.append(dist)
                robot_pairs.append((i, j))
        
        # Calculate boundary distances
        boundary_distances = []
        closest_boundary_points = []
        for i in range(4):
            nearest_point, dist = cal_nearest_point_to_boundary(
                positions[i][0], positions[i][1],
                self.origin_x_ellipse, self.origin_y_ellipse,
                self.major_axis_ellipse, self.minor_axis_ellipse,
                self.rotational_angle_ellipse
            )
            boundary_distances.append(dist)
            closest_boundary_points.append(nearest_point.flatten())
        
        # Calculate minimum distance
        robot_robot_distances = np.array(robot_robot_distances)
        boundary_distances = np.array(boundary_distances)
        min_robot_robot_dist = np.min(robot_robot_distances)
        min_boundary_dist = np.min(boundary_distances)
        min_distance = min(min_robot_robot_dist, min_boundary_dist)
        
        # Store minimum distance
        self.min_distance_values.append(min_distance)
        
        # Update visualization
        if min_distance == min_robot_robot_dist:
            min_pair_idx = np.argmin(robot_robot_distances)
            i, j = robot_pairs[min_pair_idx]
            self.min_dist_line.set_data([positions[i][0], positions[j][0]], 
                                      [positions[i][1], positions[j][1]])
        else:
            robot_idx = np.argmin(boundary_distances)
            self.min_dist_line.set_data([positions[robot_idx][0], closest_boundary_points[robot_idx][0]], 
                                      [positions[robot_idx][1], closest_boundary_points[robot_idx][1]])
        
        # self.min_dist_text.set_text(f'Min Distance: {min_distance:.2f} m')
        
        # Update robot trajectories and markers
        for i in range(4):
            self.lines[i].set_data(self.x_coords[i, :frame+1], self.y_coords[i, :frame+1])
            self.start_markers[i].set_offsets([self.x_coords[i, 0], self.y_coords[i, 0]])
            self.end_markers[i].set_offsets([self.x_coords[i, frame], self.y_coords[i, frame]])
            self.boundary_crosses[i].set_offsets([closest_boundary_points[i][0], closest_boundary_points[i][1]])
        
        # Update obstacle position
        self.obstacle_marker.set_offsets([self.x_coords[4, frame], self.y_coords[4, frame]])
        
        # Save frame
        self.save_frame(frame, positions, closest_boundary_points, min_distance, 
                       min_robot_robot_dist, robot_pairs, min_pair_idx if min_distance == min_robot_robot_dist else None,
                       robot_idx if min_distance != min_robot_robot_dist else None)
        
        time.sleep(0.5)

        return self.get_artists()
    
    def get_artists(self):
        return (self.lines + [self.obstacle_marker] + self.start_markers + 
                self.end_markers + self.boundary_crosses + 
                [self.min_dist_text, self.min_dist_line])
    
    def save_frame(self, frame, positions, closest_boundary_points, min_distance, 
                  min_robot_robot_dist, robot_pairs, min_pair_idx=None, robot_idx=None):
        save_fig = plt.figure(figsize=(8, 8))
        save_ax = save_fig.add_subplot(111)
        save_ax.set_aspect('equal')
        
        # Copy figure state
        save_ax.set_xlim(self.ax.get_xlim())
        save_ax.set_ylim(self.ax.get_ylim())
        save_ax.set_xlabel('X Coordinate (m)')
        save_ax.set_ylabel('Y Coordinate (m)')
        save_ax.grid(True)
        
        # Plot elements
        save_ax.plot(self.x_ellipse_final, self.y_ellipse_final, 'k-', linewidth=2, zorder=2, label='Boundary')
        
        # Create lists to store legend handles and labels
        robot_lines = []
        cp_points = []
        
        for i in range(4):
            # Plot robot trajectories
            line = save_ax.plot(self.x_coords[i, :frame+1], self.y_coords[i, :frame+1], 
                        marker='o', linestyle='-', color=self.colors[i], label=f'Robot {i+1}')[0]
            robot_lines.append(line)
            
            # Plot start and current positions
            save_ax.scatter(self.x_coords[i, 0], self.y_coords[i, 0], 
                          color=self.colors[i], marker='s', s=80)
            save_ax.scatter(self.x_coords[i, frame], self.y_coords[i, frame], 
                          color=self.colors[i], marker='*', s=150)
            
            # Plot closest points
            cp = save_ax.scatter(closest_boundary_points[i][0], closest_boundary_points[i][1], 
                          color=self.colors[i], marker='x', s=100, label=f'CP-R{i+1}')
            cp_points.append(cp)
        
        # Plot obstacle
        obstacle = save_ax.scatter(self.x_coords[4, frame], self.y_coords[4, frame], 
                       color='red', marker='D', s=80, label='Obstacle')
        
        # Plot minimum distance line
        min_dist_line = None
        if min_distance == min_robot_robot_dist and min_pair_idx is not None:
            i, j = robot_pairs[min_pair_idx]
            min_dist_line = save_ax.plot([positions[i][0], positions[j][0]], 
                                       [positions[i][1], positions[j][1]], 
                                       'k--', linewidth=2, label='Min Distance')[0]
        elif robot_idx is not None:
            min_dist_line = save_ax.plot([positions[robot_idx][0], closest_boundary_points[robot_idx][0]], 
                                       [positions[robot_idx][1], closest_boundary_points[robot_idx][1]], 
                                       'k--', linewidth=2, label='Min Distance')[0]
        # # Min distance on the figure
        # save_ax.text(0.05, 0.95, f'Min Distance: {min_distance:.2f} m', 
        #             transform=save_ax.transAxes, fontsize=12, verticalalignment='top')
        
        # Create boundary line with explicit label
        boundary_line = save_ax.get_lines()[0]
        boundary_line.set_label('Boundary')
        
        # Organize legend handles in desired order - use list instead of tuple
        legend_handles = robot_lines + cp_points + [obstacle, boundary_line]
        
        # Add min distance line to legend if it exists
        if min_dist_line is not None:
            legend_handles.append(min_dist_line)
            
        # Filter out artists with labels starting with underscore
        legend_handles = [art for art in legend_handles if not art.get_label().startswith('_')]
        
        save_ax.legend(handles=legend_handles, loc='upper right')
        
        # Save files
        plt.savefig(f'graph/frame_field_{frame:03d}.pdf', format='pdf', dpi=300, 
                   bbox_inches='tight', pad_inches=0.1)
        plt.savefig(f'graph/frame_field_{frame:03d}.png', format='png', dpi=300, 
                   bbox_inches='tight', pad_inches=0.1)
        
        plt.close(save_fig)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min(np.min(x_coords), np.min(x_ellipse_final)) - 5, max(np.max(x_coords), np.max(x_ellipse_final)) + 5)
ax.set_ylim(min(np.min(y_coords), np.min(y_ellipse_final)) - 5, max(np.max(y_coords), max(y_ellipse_final)) + 5)
ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.grid(True)
ax.set_aspect('equal')

# Plot the ellipse
ax.plot(x_ellipse_final, y_ellipse_final, 'k-', linewidth=2, zorder=2)

# Initialize empty plots for each robot
lines = [ax.plot([], [], marker='o', linestyle='-', color=colors[i], label=f'Robot {i+1}')[0] for i in range(4)]
obstacle_marker = ax.scatter([], [], color='red', marker='H', s=100, label='Obstacle')

start_markers = [ax.scatter([], [], color=colors[i], marker='s', s=80) for i in range(5)]
end_markers = [ax.scatter([], [], color=colors[i], marker='*', s=150) for i in range(5)]

# Text annotation for minimum distance
min_dist_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Line to indicate minimum distance
min_dist_line, = ax.plot([], [], 'k--', linewidth=2)

# Initialize boundary crossing markers
boundary_crosses = [ax.scatter([], [], color=colors[i], marker='x', s=100, label=f'CP-R{i+1}') for i in range(4)]

# Create ordered legend items
obstacle_legend = Line2D([0], [0], marker='H', color='w', markerfacecolor='red', markersize=8, label='Obstacle')
boundary_legend = Line2D([0], [0], linestyle='-', color='black', linewidth=2, label='Boundary')
min_distance_legend = Line2D([0], [0], linestyle='--', color='black', linewidth=2, label='Min Distance')

# Create legend handles list
legend_handles = lines + boundary_crosses + [obstacle_legend, boundary_legend, min_distance_legend]

# Filter out artists with labels starting with underscore
legend_handles = [art for art in legend_handles if not art.get_label().startswith('_')]

# Reorder legend items: Robots (1-4) first, then CP-Rs, then others
ax.legend(loc='upper right', handles=legend_handles)

# Create animation manager
animation_manager = AnimationManager(fig, ax, x_coords, y_coords, x_ellipse_final, 
                                  y_ellipse_final, origin_x_ellipse, origin_y_ellipse, 
                                  major_axis_ellipse, minor_axis_ellipse, 
                                  rotational_angle_ellipse)

# Create animation
ani = animation.FuncAnimation(fig, animation_manager.update, frames=range(num_frames), 
                            interval=500, blit=True, repeat=False)
plt.show()

# Save results
min_distance_values = animation_manager.min_distance_values
min_distance_array = np.array(min_distance_values)
# np.save("graph/min_distance_sim.npy", min_distance_array)

print('len(min_distance_values): ', len(min_distance_values))
time_values = time_val # np.arange(len(min_distance_values)) * time_factor
df = pd.DataFrame({"Time": time_values, "Min Distance": min_distance_values})
# df.to_csv("graph/min_distance_sim.csv", index=False)

# Create final plots
plt.figure(figsize=(10, 5))
plt.plot(df["Time"], df["Min Distance"], marker="o", linestyle="-")
plt.xlabel("Time (s)")
plt.ylabel("Min Distance (m)")
plt.grid(True)
plt.xlim(0, max(time_values))
plt.ylim(0, max(df["Min Distance"]) * 1.1)

plt.savefig('graph/minimum_distance_plot_field.pdf', format='pdf', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.savefig('graph/minimum_distance_plot_field.png', format='png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

