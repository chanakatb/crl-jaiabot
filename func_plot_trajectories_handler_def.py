import os
import signal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ProgramController:
    def __init__(self):
        self.running = True
        self.current_count = 0
        self.current_x = None
        
    def update_state(self, x, count):
        self.current_count = count
        self.current_x = x

def create_signal_handler(controller, origin_x_ellipse, origin_y_ellipse, 
                         major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse, 
                         min_index_registry, robots_lat_lon_arr, hub_lat_lon_arr,
                         no_of_robots, no_of_obstacles, timestamp):
    def signal_handler(signum, frame):
        print("\nStopping execution...")
        
        plotter = TrajectoryPlotter(
            controller.current_x, 
            controller.current_count,
            origin_x_ellipse, origin_y_ellipse,
            major_axis_ellipse, minor_axis_ellipse,
            rotational_angle_ellipse,
            no_of_robots,
            no_of_obstacles,
            timestamp
        )
        
        plotter.create_plot()  # Removed timestamp return since we're using the input timestamp
        plotter.save_plot()    # Removed timestamp parameter since it's stored in the class
        controller.running = False
        
    return signal_handler

class TrajectoryPlotter:
    def __init__(self, x, count, origin_x_ellipse, origin_y_ellipse, 
                 major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse, 
                 no_of_robots, no_of_obstacles, timestamp):
        """
        Args:
            x: State array of shape (2, total_entities, timesteps)
               where total_entities includes robots, obstacles, and boundary points
            count: Current timestep
            no_of_robots: Number of robots
            no_of_obstacles: Number of obstacles
            timestamp: Timestamp string for file naming
        """
        # Store robot positions
        self.x = x[:, :no_of_robots, :]  # Take the first no_of_robots columns for robots
        
        # Store obstacle positions (take final positions only)
        self.obstacles_position = x[:, no_of_robots:(no_of_robots + no_of_obstacles), -1]
        
        self.count = count
        self.origin_x_ellipse = origin_x_ellipse
        self.origin_y_ellipse = origin_y_ellipse
        self.major_axis_ellipse = major_axis_ellipse
        self.minor_axis_ellipse = minor_axis_ellipse
        self.rotational_angle_ellipse = rotational_angle_ellipse
        self.no_of_robots = no_of_robots
        self.no_of_obstacles = no_of_obstacles
        self.timestamp = timestamp  # Store timestamp as class attribute

    def create_plot(self):       
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot robot trajectories
        colors = ['blue', 'green', 'orange', 'purple']
        lines = []  # Store line objects for legend
        
        # Plot trajectories for each robot
        for i in range(self.no_of_robots):
            # Plot full trajectory with small dots
            trajectory_line = plt.plot(self.x[0, i, :self.count+1], 
                                     self.x[1, i, :self.count+1], 
                                     '.', color=colors[i], markersize=2,
                                     zorder=1)[0]
            
            # Create legend line
            legend_line = plt.plot([], [], 'o', color=colors[i], 
                                 label=f'Robot {i+1}', markersize=15)[0]
            lines.append(legend_line)
            
            # Plot start position
            plt.plot(self.x[0, i, 0], self.x[1, i, 0], 'o', 
                    color=colors[i], markersize=12, 
                    markerfacecolor=colors[i], 
                    markeredgecolor='black', markeredgewidth=1.5,
                    zorder=2)
            
            # Plot end positions
            plt.plot(self.x[0, i, self.count], self.x[1, i, self.count], 
                    '*', color=colors[i], markersize=15, markeredgewidth=1.5,
                    zorder=2)
        
        # Add 'Start' labels for robots
        for i in range(self.no_of_robots):
            plt.annotate('Start', 
                        (self.x[0, i, 0], self.x[1, i, 0]),
                        xytext=(10, 5),
                        textcoords='offset points',
                        fontsize=10,
                        color=colors[i])
        
        # Plot obstacles
        if self.no_of_obstacles > 0:
            # Plot obstacles with red X markers
            obstacles_scatter = plt.scatter(self.obstacles_position[0], 
                                         self.obstacles_position[1],
                                         c='red', marker='x', s=100, 
                                         label='Obstacles', zorder=3)
            
            # Add labels for obstacles
            for i in range(self.no_of_obstacles):
                plt.annotate(f'Obs {i+1}',
                           (self.obstacles_position[0, i], self.obstacles_position[1, i]),
                           xytext=(5, 5),
                           textcoords='offset points',
                           color='red',
                           fontsize=10,
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Create and plot ellipse
        self._plot_ellipse()
        
        # Add hub point
        hub_point = plt.plot(0, 0, 'ks', label='Hub', markersize=10, zorder=2)[0]
        
        # Set plot properties
        plt.grid(True)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Trajectories, Obstacles, and Boundary')
        plt.axis('equal')
        
        # Create legend with obstacles
        legend_elements = [*lines]
        if self.no_of_obstacles > 0:
            legend_elements.append(obstacles_scatter)
        legend_elements.extend([self.boundary_line, hub_point])
        
        plt.legend(handles=legend_elements, labelspacing=1, markerscale=1)
    
    def _plot_ellipse(self):
        # Create ellipse points
        t = np.linspace(0, 2*np.pi, 200)
        ell_x = self.major_axis_ellipse * np.cos(t)
        ell_y = self.minor_axis_ellipse * np.sin(t)
        
        # Rotate ellipse
        x_rot = (ell_x * np.cos(self.rotational_angle_ellipse) - 
                ell_y * np.sin(self.rotational_angle_ellipse))
        y_rot = (ell_x * np.sin(self.rotational_angle_ellipse) + 
                ell_y * np.cos(self.rotational_angle_ellipse))
        
        # Translate ellipse
        x_rot += self.origin_x_ellipse
        y_rot += self.origin_y_ellipse
        
        # Plot ellipse
        self.boundary_line = plt.plot(x_rot, y_rot, 'k-', 
                                    label='Boundary', linewidth=2)[0]
    
    def save_plot(self):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save plots using the stored timestamp
        plt.savefig(os.path.join('data', f'{self.timestamp}_trajectories.png'))
        # plt.savefig(os.path.join('data', f'trajectories_{self.timestamp}.pdf'))
        
        # Display the plot
        plt.show()
        
        # Print save information
        print(f"\nData and plots saved with timestamp: {self.timestamp}")
        print(f"Files saved in 'data' directory:")
        print(f"{self.timestamp}_trajectories.png")
        # print(f"{self.timestamp}_trajectories.pdf")