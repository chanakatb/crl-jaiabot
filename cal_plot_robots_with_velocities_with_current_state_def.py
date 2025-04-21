import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.ion()  # Enable interactive mode

class RobotVisualizer:
    def __init__(self, figsize_robots=(12, 10), figsize_distance=(10, 6)):
        """Initialize the visualizer with two separate figures"""
        # Create main robot position figure
        self.fig_robots = plt.figure(figsize=figsize_robots)
        self.ax_robots = self.fig_robots.add_subplot(111)
        
        # Create separate distance figure
        self.fig_distance = plt.figure(figsize=figsize_distance)
        self.ax_distance = self.fig_distance.add_subplot(111)
        
        # Initialize distance tracking
        self.distances = []
        self.timestamps = []
        self.current_time = 0
        
        # Define colors for each robot
        self.robot_colors = {
            1: 'blue',
            2: 'green',
            3: 'orange',
            4: 'purple'
        }
        
        # Initialize empty scatter plots for each robot
        self.scatters = {}
        for bot_id, color in self.robot_colors.items():
            self.scatters[bot_id] = self.ax_robots.scatter([], [], color=color, s=100, label=f'Bot {bot_id}')
            
        # Initialize obstacles scatter plot
        self.obstacles_scatter = self.ax_robots.scatter([], [], color='red', marker='x', s=100, label='Obstacles')
        
        # Initialize storage for plot elements
        self.arrows = []
        self.texts = []
        self.bot_labels = []
        self.min_distance_line = None
        self.ellipse = None
        self.hub_marker = None
        
        # Create proper arrow patches for the legend
        target_arrow_legend = FancyArrowPatch((0, 0), (1, 0),
                                            color='red',
                                            mutation_scale=15)
        current_arrow_legend = FancyArrowPatch((0, 0), (1, 0),
                                            color='green',
                                            mutation_scale=15)
        self.min_dist_line = plt.Line2D([], [], color='purple', linestyle='--', label='Min Distance')
        self.boundary_line = plt.Line2D([], [], color='black', linestyle='-', label='Boundary')
        
        # Create legend with all elements
        legend_elements = [
            *[self.scatters[bot_id] for bot_id in sorted(self.scatters.keys())],
            target_arrow_legend,
            current_arrow_legend,
            self.min_dist_line,
            self.boundary_line,
            self.obstacles_scatter
        ]
        legend_labels = [
            *[f'Bot {bot_id}' for bot_id in sorted(self.scatters.keys())],
            'Target State',
            'Current State',
            'Min Distance',
            'Boundary',
            'Obstacles'
        ]
        
        self.ax_robots.legend(legend_elements, legend_labels, loc='upper right')
        
        # Setup robot position plot
        self.ax_robots.grid(True)
        self.ax_robots.set_xlabel('X Position (m)')
        self.ax_robots.set_ylabel('Y Position (m)')
        self.ax_robots.set_title('Robots Position with Target State')
        
        # Setup distance plot
        self.ax_distance.grid(True)
        self.ax_distance.set_xlabel('Iteration')
        self.ax_distance.set_ylabel('Minimum Distance (m)')
        self.ax_distance.set_title('Minimum Distance Over Time')
        
        # Show both figures
        self.fig_robots.show()
        self.fig_distance.show()
    
    def update(self, bot_ids, robots_position, obstacles_position, bot_data, robots_current_heading, robots_current_speed, min_distance_coordinates=None, boundary_parameters=None):
        """Update both plots with new data"""
        # Clear previous elements in robot plot
        for arrow in self.arrows:
            arrow.remove()
        for text in self.texts:
            text.remove()
        for label in self.bot_labels:
            label.remove()
        if self.min_distance_line is not None:
            self.min_distance_line.remove()
        if hasattr(self, 'ellipse') and self.ellipse is not None:
            self.ellipse.remove()
        if hasattr(self, 'hub_marker') and self.hub_marker is not None:
            self.hub_marker.remove()
        
        self.arrows = []
        self.texts = []
        self.bot_labels = []
        
        # Update obstacles positions
        if obstacles_position is not None and obstacles_position.shape[1] > 0:
            self.obstacles_scatter.set_offsets(np.column_stack((obstacles_position[0], obstacles_position[1])))
            
            # Add labels for obstacles
            for i in range(obstacles_position.shape[1]):
                obstacle_label = self.ax_robots.annotate(f'Obs {i+1}', 
                                                     (obstacles_position[0][i], obstacles_position[1][i]),
                                                     xytext=(5, 5),
                                                     textcoords='offset points',
                                                     bbox=dict(boxstyle='round,pad=0.2', 
                                                             fc='red',
                                                             alpha=0.3),
                                                     color='black',
                                                     fontweight='bold',
                                                     ha='left',
                                                     va='bottom')
                self.texts.append(obstacle_label)
        else:
            # Clear obstacles if none present
            self.obstacles_scatter.set_offsets(np.empty((0, 2)))
        
        # Draw hub marker at (0,0)
        self.hub_marker = self.ax_robots.scatter(0, 0, marker='s', s=100, color='black', label='Hub')
        hub_text = self.ax_robots.annotate('Hub', 
                                       (0, 0), 
                                       xytext=(10, 10),
                                       textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        self.texts.append(hub_text)
        
        # Draw elliptical boundary if parameters are provided
        if boundary_parameters is not None:
            origin_x, origin_y, major_axis, minor_axis, angle = boundary_parameters
            from matplotlib.patches import Ellipse
            angle_deg = np.degrees(angle)
            self.ellipse = Ellipse(xy=(origin_x, origin_y), 
                                width=major_axis*2, 
                                height=minor_axis*2,
                                angle=angle_deg,
                                fill=False, 
                                color='black', 
                                linestyle='-',
                                linewidth=2)
            self.ax_robots.add_patch(self.ellipse)
        
        # Scale factors for arrows
        heading_scale = 10.0
        y_start = 0.7
        text_spacing = 0.1
        
        # Update each robot's data
        for i, bot_id in enumerate(bot_ids):
            x = robots_position[0][i]
            y = robots_position[1][i]
            
            # Update scatter positions
            self.scatters[bot_id].set_offsets(np.array([[x, y]]))
            
            # Add bot number labels
            bot_label = self.ax_robots.annotate(f'{bot_id}', 
                                            (x, y),
                                            xytext=(5, 5),
                                            textcoords='offset points',
                                            bbox=dict(boxstyle='circle,pad=0.5', 
                                                    fc=self.robot_colors[bot_id],
                                                    alpha=0.3),
                                            color='black',
                                            fontweight='bold',
                                            ha='left',
                                            va='bottom')
            self.bot_labels.append(bot_label)
            
            # Target velocity vector
            target_speed = bot_data[i]['speed']
            target_heading_rad = np.radians(bot_data[i]['heading'])
            target_x = np.cos(np.radians(90) - target_heading_rad) * target_speed
            target_y = np.sin(np.radians(90) - target_heading_rad) * target_speed
            
            # Draw target state arrow (red)
            target_arrow = self.ax_robots.arrow(x, y, 
                                            target_x * heading_scale, 
                                            target_y * heading_scale,
                                            head_width=1.0, 
                                            head_length=1.5,
                                            fc='red', 
                                            ec='red', 
                                            alpha=0.6)
            self.arrows.append(target_arrow)
            
            # Current velocity vector
            current_speed = robots_current_speed[0][i]
            current_heading_rad = np.radians(robots_current_heading[0][i])
            current_x = np.cos(np.radians(90) - current_heading_rad) * current_speed
            current_y = np.sin(np.radians(90) - current_heading_rad) * current_speed
            
            # Draw current state arrow (green)
            current_arrow = self.ax_robots.arrow(x, y, 
                                             current_x * heading_scale, 
                                             current_y * heading_scale,
                                             head_width=1.0, 
                                             head_length=1.5,
                                             fc='green', 
                                             ec='green', 
                                             alpha=0.6)
            self.arrows.append(current_arrow)
            
            # Update text information to include current state
            text = self.ax_robots.text(
                0.95, y_start - (i * text_spacing),
                f'Bot {bot_id}\n'
                f'X: {x:.1f}, Y: {y:.1f}\n'
                f'Target: {bot_data[i]["speed"]:.2f} m/s, {bot_data[i]["heading"]:.1f}°\n'
                f'Current: {current_speed:.2f} m/s, {robots_current_heading[0][i]:.1f}°',
                transform=self.ax_robots.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                ha='right',
                va='top'
            )
            self.texts.append(text)
        
        # Update minimum distance data and plots
        if min_distance_coordinates is not None:
            dx = min_distance_coordinates[0][1] - min_distance_coordinates[0][0]
            dy = min_distance_coordinates[1][1] - min_distance_coordinates[1][0]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Update distance history
            self.distances.append(distance)
            self.timestamps.append(self.current_time)
            
            # Draw minimum distance line on robot plot
            self.min_distance_line = self.ax_robots.plot(
                [min_distance_coordinates[0][0], min_distance_coordinates[0][1]],
                [min_distance_coordinates[1][0], min_distance_coordinates[1][1]],
                'purple', linestyle='--', linewidth=2, alpha=0.8
            )[0]
            
            # Add distance label
            mid_x = (min_distance_coordinates[0][0] + min_distance_coordinates[0][1]) / 2
            mid_y = (min_distance_coordinates[1][0] + min_distance_coordinates[1][1]) / 2
            text = self.ax_robots.annotate(
                f'{distance:.1f} m',
                (mid_x, mid_y), xytext=(0, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                ha='center',
                va='bottom'
            )
            self.texts.append(text)
            
            # Update distance plot
            self.ax_distance.clear()
            self.ax_distance.plot(self.timestamps, self.distances, 'b-o', 
                                markerfacecolor='blue',
                                markeredgecolor='darkblue',
                                markeredgewidth=1.5,
                                markersize=8,
                                linewidth=2)

            # Add statistics
            if len(self.distances) > 0:
                min_dist = min(self.distances)
                max_dist = max(self.distances)
                avg_dist = sum(self.distances) / len(self.distances)
                
                stats_text = f'Min: {min_dist:.1f}m'
                self.ax_distance.text(0.02, 0.98, stats_text,
                                    transform=self.ax_distance.transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

                current_text = f'Current Min Distance: {distance:.1f}m'
                self.ax_distance.text(0.98, 0.98, current_text,
                                    transform=self.ax_distance.transAxes,
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.5', 
                                            fc='white', 
                                            ec='black',
                                            alpha=0.7))

            self.ax_distance.grid(True)
            self.ax_distance.set_xlabel('Iteration')
            self.ax_distance.set_ylabel('Minimum Distance (m)')
            self.ax_distance.set_title('Minimum Distance Over Time')
        
        # Update plot limits and refresh
        self.ax_robots.relim()
        self.ax_robots.autoscale_view()
        self.ax_robots.axis('equal')
        
        # Increment time
        self.current_time += 1
        
        # Redraw both plots
        self.fig_robots.canvas.draw()
        self.fig_robots.canvas.flush_events()
        self.fig_distance.canvas.draw()
        self.fig_distance.canvas.flush_events()