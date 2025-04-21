import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.ion()  # Enable interactive mode

class RobotVisualizer_predicted:
    def __init__(self, bot_ids=None, figsize_robots=(12, 10)):
        # Create main robot position figure
        self.fig_robots = plt.figure(figsize=figsize_robots)
        self.ax_robots = self.fig_robots.add_subplot(111)
        
        # Initialize distance tracking
        self.current_time = 0
        
        # Set default bot_ids if none provided
        if bot_ids is None:
            bot_ids = [1, 2, 3, 4]
        
        # Define colors for each robot (ensure enough colors for all bots)
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
        self.robot_colors = {}
        for i, bot_id in enumerate(bot_ids):
            self.robot_colors[bot_id] = colors[i % len(colors)]
        
        # Initialize empty scatter plots for each robot
        self.scatters = {}
        self.predicted_scatters = {}
        self.prediction_lines = {}
        
        for bot_id in bot_ids:
            color = self.robot_colors[bot_id]
            self.scatters[bot_id] = self.ax_robots.scatter([], [], color=color, s=100, label=f'Robot {bot_id}')
            self.predicted_scatters[bot_id] = self.ax_robots.scatter([], [], color=color, marker='o', s=50, alpha=0.5)
            self.prediction_lines[bot_id] = None    

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
        target_arrow_legend = plt.arrow(0, 0, 1, 0, color='red', width=0.1, head_width=0.3, 
                                      head_length=0.3, length_includes_head=True)
        current_arrow_legend = plt.arrow(0, 0, 1, 0, color='green', width=0.1, head_width=0.3, 
                                       head_length=0.3, length_includes_head=True)
        phi_grad_arrow_legend = plt.arrow(0, 0, 1, 0, color='blue', width=0.1, head_width=0.3, 
                                        head_length=0.3, length_includes_head=True)
        
        # Create legend elements for arrows properly
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Create custom legend handles
        custom_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Target Heading'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Current Heading'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Phi Gradient')
        ]
        
        # Create other legend elements
        prediction_line_legend = plt.Line2D([], [], color='gray', linestyle=':', label='Predicted Path')
        self.min_dist_line = plt.Line2D([], [], color='purple', linestyle='--', label='Min Distance')
        self.boundary_line = plt.Line2D([], [], color='black', linestyle='-', label='Boundary')
        
        # Create legend with all elements
        legend_elements = [
            *[self.scatters[bot_id] for bot_id in sorted(self.scatters.keys())],
            Line2D([0], [0], color='red', lw=2, marker='>', markersize=10, label='Target Heading'),
            Line2D([0], [0], color='green', lw=2, marker='>', markersize=10, label='Current Heading'),
            Line2D([0], [0], color='blue', lw=2, marker='>', markersize=10, label='Phi Gradient'),
            prediction_line_legend,
            self.min_dist_line,
            self.boundary_line,
            self.obstacles_scatter
        ]
        
        # Move legend to upper left corner
        self.ax_robots.legend(handles=legend_elements, loc='upper left')
        
        # Setup robot position plot
        self.ax_robots.grid(True)
        self.ax_robots.set_xlabel('X Position (m)')
        self.ax_robots.set_ylabel('Y Position (m)')
        self.ax_robots.set_title("Multi-Robot Navigation with Potential Field Gradients")
        
        # Show figure
        self.fig_robots.show()
    
    def update(self, bot_ids, robots_position, obstacles_position, robots_current_heading, 
              robots_current_speed, coords_predicted, min_distance_coordinates=None, boundary_parameters=None, phi_grad_arrows=None):
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
            
        # Remove previous prediction lines
        for bot_id in self.prediction_lines:
            if self.prediction_lines[bot_id] is not None:
                self.prediction_lines[bot_id].remove()
                self.prediction_lines[bot_id] = None
        
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
        # Scale factor for target heading arrow - making it 1.5x the distance
        target_scale_factor = 1.5
        
        # Position the text boxes at the right side, starting from the top
        y_start = 0.95  # Start higher up
        text_spacing = 0.15
        
        # Update each robot's data
        for i, bot_id in enumerate(bot_ids):
            # Ensure coordinates are scalar floats
            x = float(robots_position[0][i])
            y = float(robots_position[1][i])
            
            # Get predicted coordinates for this bot and ensure they're scalar floats
            pred_x = float(coords_predicted[2*i])
            pred_y = float(coords_predicted[2*i + 1])
            
            # Update scatter positions for current and predicted positions
            self.scatters[bot_id].set_offsets(np.array([[x, y]]))
            self.predicted_scatters[bot_id].set_offsets(np.array([[pred_x, pred_y]]))
            
            # Draw prediction path line - ensuring all values are scalar floats
            self.prediction_lines[bot_id] = self.ax_robots.plot([x, pred_x], [y, pred_y], 
                                                          color=self.robot_colors[bot_id],
                                                          linestyle=':', 
                                                          linewidth=1.5,
                                                          alpha=0.6)[0]
            
            # Add robot number labels
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
            
            # Calculate vector between current and predicted position
            target_x = pred_x - x
            target_y = pred_y - y
            
            # Calculate the distance between current and predicted position
            distance = np.sqrt(target_x**2 + target_y**2)
            
            # Calculate target heading length - 1.5 times the actual distance
            target_heading_length = 1.5 * distance
            
            # Calculate target arrow components with the 1.5x scaling
            if distance > 0:
                # Scale the target vector to 1.5 times the distance
                scaled_x = target_x * target_scale_factor
                scaled_y = target_y * target_scale_factor
            else:
                # If distance is 0, use a small default value to avoid division by zero
                scaled_x = 0
                scaled_y = 0
            
            # Draw target heading arrow (red) with 1.5x length
            target_arrow = self.ax_robots.arrow(x, y, 
                                        scaled_x, 
                                        scaled_y,
                                        head_width=1.0, 
                                        head_length=1.5,
                                        fc='red', 
                                        ec='red', 
                                        alpha=0.6)
            self.arrows.append(target_arrow)
            
            # Default values for gradient information
            grad_x = 0
            grad_y = 0
            grad_magnitude = 0
            
            # Draw phi gradient arrow if provided
            if phi_grad_arrows is not None and i < phi_grad_arrows.shape[1]:
                # Get the gradient components for this bot
                grad_x = float(phi_grad_arrows[0, i])
                grad_y = float(phi_grad_arrows[1, i])
                
                # Calculate magnitude
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Only draw if magnitude is not too small
                if grad_magnitude > 1e-6:
                    # Make gradient arrow same length as target heading arrow (target_heading_length)
                    if grad_magnitude > 0:
                        # Scale gradient to match the target heading length
                        scale_factor = target_heading_length / grad_magnitude
                        scaled_grad_x = grad_x * scale_factor
                        scaled_grad_y = grad_y * scale_factor
                    else:
                        # Avoid division by zero
                        scaled_grad_x = 0
                        scaled_grad_y = 0
                    
                    # Draw gradient arrow (blue) with same length as target arrow
                    grad_arrow = self.ax_robots.arrow(x, y, 
                                            scaled_grad_x, 
                                            scaled_grad_y,
                                            head_width=1.0, 
                                            head_length=1.5,
                                            fc='blue', 
                                            ec='blue', 
                                            alpha=0.7)
                    self.arrows.append(grad_arrow)
            
            # Current velocity vector - ensure scalar values
            current_speed = float(robots_current_speed[0][i])
            current_heading_rad = np.radians(float(robots_current_heading[0][i]))
            current_x = np.cos(np.radians(90) - current_heading_rad) * current_speed
            current_y = np.sin(np.radians(90) - current_heading_rad) * current_speed
            
            # Draw current heading arrow (green)
            current_arrow = self.ax_robots.arrow(x, y, 
                                         current_x * heading_scale, 
                                         current_y * heading_scale,
                                         head_width=1.0, 
                                         head_length=1.5,
                                         fc='green', 
                                         ec='green', 
                                         alpha=0.6)
            self.arrows.append(current_arrow)
            
            # Update text information including next waypoint distance and gradient info
            text = self.ax_robots.text(
                0.99, y_start - (i * text_spacing),  # Position at right edge
                f'$\\mathbf{{Robot\\ {bot_id}}}$\n'  # Bold robot name using LaTeX math syntax
                f'Current: X: {x:.1f}, Y: {y:.1f}\n'
                f'Predicted: X: {pred_x:.1f}, Y: {pred_y:.1f}\n'
                f'Next Waypoint: {distance:.2f} m\n'
                f'Current Speed: {current_speed:.2f} m/s\n'
                f'Current Heading: {robots_current_heading[0][i]:.1f}°\n'
                f'Gradient: X: {grad_x:.4f}, Y: {grad_y:.4f}, Mag: {grad_magnitude:.4f}',
                transform=self.ax_robots.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                ha='right',  # Right-align the text
                va='top'  # Align to top
            )
            self.texts.append(text)

        # Update minimum distance line
        if min_distance_coordinates is not None:
            dx = float(min_distance_coordinates[0][1]) - float(min_distance_coordinates[0][0])
            dy = float(min_distance_coordinates[1][1]) - float(min_distance_coordinates[1][0])
            distance = np.sqrt(dx**2 + dy**2)
            
            # Ensure scalar float values for plotting
            x0 = float(min_distance_coordinates[0][0])
            y0 = float(min_distance_coordinates[1][0])
            x1 = float(min_distance_coordinates[0][1])
            y1 = float(min_distance_coordinates[1][1])
            
            self.min_distance_line = self.ax_robots.plot(
                [x0, x1],
                [y0, y1],
                'purple', linestyle='--', linewidth=2, alpha=0.8
            )[0]
            
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            text = self.ax_robots.annotate(
                f'{distance:.1f} m',
                (mid_x, mid_y), xytext=(0, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                ha='center',
                va='bottom'
            )
            self.texts.append(text)
        
        # Update plot limits and refresh
        self.ax_robots.relim()
        self.ax_robots.autoscale_view()
        self.ax_robots.axis('equal')
        
        # Increment time
        self.current_time += 1
        
        # Redraw the plot
        self.fig_robots.canvas.draw()
        self.fig_robots.canvas.flush_events()


"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.ion()  # Enable interactive mode

class RobotVisualizer_predicted:
    def __init__(self, bot_ids=None, figsize_robots=(12, 10)):
        # Create main robot position figure
        self.fig_robots = plt.figure(figsize=figsize_robots)
        self.ax_robots = self.fig_robots.add_subplot(111)
        
        # Initialize distance tracking
        self.current_time = 0
        
        # Set default bot_ids if none provided
        if bot_ids is None:
            bot_ids = [1, 2, 3, 4]
        
        # Define colors for each robot (ensure enough colors for all bots)
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
        self.robot_colors = {}
        for i, bot_id in enumerate(bot_ids):
            self.robot_colors[bot_id] = colors[i % len(colors)]
        
        # Initialize empty scatter plots for each robot
        self.scatters = {}
        self.predicted_scatters = {}
        self.prediction_lines = {}
        
        for bot_id in bot_ids:
            color = self.robot_colors[bot_id]
            self.scatters[bot_id] = self.ax_robots.scatter([], [], color=color, s=100, label=f'Robot {bot_id}')
            self.predicted_scatters[bot_id] = self.ax_robots.scatter([], [], color=color, marker='o', s=50, alpha=0.5)
            self.prediction_lines[bot_id] = None    

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
        prediction_line_legend = plt.Line2D([], [], color='gray', linestyle=':', label='Predicted Path')
        self.min_dist_line = plt.Line2D([], [], color='purple', linestyle='--', label='Min Distance')
        self.boundary_line = plt.Line2D([], [], color='black', linestyle='-', label='Boundary')
        
        # Create legend with all elements
        legend_elements = [
            *[self.scatters[bot_id] for bot_id in sorted(self.scatters.keys())],
            target_arrow_legend,
            current_arrow_legend,
            prediction_line_legend,
            self.min_dist_line,
            self.boundary_line,
            self.obstacles_scatter
        ]
        legend_labels = [
            # Changed bot labels to robot labels
            *[f'Robot {bot_id}' for bot_id in sorted(self.scatters.keys())],
            'Target Heading',
            'Current Heading',
            'Predicted Path',
            'Min Distance',
            'Boundary',
            'Obstacles'
        ]
        
        # Move legend to upper left corner
        self.ax_robots.legend(legend_elements, legend_labels, loc='upper left')
        
        # Setup robot position plot
        self.ax_robots.grid(True)
        self.ax_robots.set_xlabel('X Position (m)')
        self.ax_robots.set_ylabel('Y Position (m)')
        self.ax_robots.set_title("Robots' Current and Predicted Positions")
        
        # Show figure
        self.fig_robots.show()
    
    def update(self, bot_ids, robots_position, obstacles_position, robots_current_heading, 
              robots_current_speed, coords_predicted, min_distance_coordinates=None, boundary_parameters=None):
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
            
        # Remove previous prediction lines
        for bot_id in self.prediction_lines:
            if self.prediction_lines[bot_id] is not None:
                self.prediction_lines[bot_id].remove()
                self.prediction_lines[bot_id] = None
        
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
        
        # Position the text boxes at the right side, starting from the top
        y_start = 0.95  # Start higher up
        text_spacing = 0.15
        
        # Update each robot's data
        for i, bot_id in enumerate(bot_ids):
            x = robots_position[0][i]
            y = robots_position[1][i]
            
            # Get predicted coordinates for this bot
            pred_x = coords_predicted[2*i]
            pred_y = coords_predicted[2*i + 1]
            
            # Update scatter positions for current and predicted positions
            self.scatters[bot_id].set_offsets(np.array([[x, y]]))
            self.predicted_scatters[bot_id].set_offsets(np.array([[pred_x, pred_y]]))
            
            # Draw prediction path line
            self.prediction_lines[bot_id] = self.ax_robots.plot([x, pred_x], [y, pred_y], 
                                                          color=self.robot_colors[bot_id],
                                                          linestyle=':', 
                                                          linewidth=1.5,
                                                          alpha=0.6)[0]
            
            # Add robot number labels (changed from bot to robot)
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
            
            # Calculate vector between current and predicted position
            target_x = pred_x - x
            target_y = pred_y - y
            
            # Calculate the distance between current and predicted position
            distance = np.sqrt(target_x**2 + target_y**2)
            
            # Scale the target heading arrow to 1.5 times the distance
            if distance > 0:
                scale_factor = 1.5
                scaled_x = target_x * scale_factor
                scaled_y = target_y * scale_factor
            else:
                # If distance is 0, use a small default value to avoid division by zero
                scaled_x = 0
                scaled_y = 0
            
            # Draw target heading arrow (red)
            target_arrow = self.ax_robots.arrow(x, y, 
                                        scaled_x, 
                                        scaled_y,
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
            
            # Draw current heading arrow (green)
            current_arrow = self.ax_robots.arrow(x, y, 
                                         current_x * heading_scale, 
                                         current_y * heading_scale,
                                         head_width=1.0, 
                                         head_length=1.5,
                                         fc='green', 
                                         ec='green', 
                                         alpha=0.6)
            self.arrows.append(current_arrow)
            
            # Update text information including next waypoint distance - moved to right side
            text = self.ax_robots.text(
                0.99, y_start - (i * text_spacing),  # Position at right edge
                f'$\\mathbf{{Robot\\ {bot_id}}}$\n'  # Bold robot name using LaTeX math syntax
                f'Current: X: {x:.1f}, Y: {y:.1f}\n'
                f'Predicted: X: {pred_x:.1f}, Y: {pred_y:.1f}\n'
                f'Next Waypoint: {distance:.2f} m\n'  # Added waypoint distance
                f'Current Speed: {current_speed:.2f} m/s\n'
                f'Current Heading: {robots_current_heading[0][i]:.1f}°',
                transform=self.ax_robots.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                ha='right',  # Right-align the text
                va='top'  # Align to top
            )
            self.texts.append(text)

        # Update minimum distance line
        if min_distance_coordinates is not None:
            dx = min_distance_coordinates[0][1] - min_distance_coordinates[0][0]
            dy = min_distance_coordinates[1][1] - min_distance_coordinates[1][0]
            distance = np.sqrt(dx**2 + dy**2)
            
            self.min_distance_line = self.ax_robots.plot(
                [min_distance_coordinates[0][0], min_distance_coordinates[0][1]],
                [min_distance_coordinates[1][0], min_distance_coordinates[1][1]],
                'purple', linestyle='--', linewidth=2, alpha=0.8
            )[0]
            
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
        
        # Update plot limits and refresh
        self.ax_robots.relim()
        self.ax_robots.autoscale_view()
        self.ax_robots.axis('equal')
        
        # Increment time
        self.current_time += 1
        
        # Redraw the plot
        self.fig_robots.canvas.draw()
        self.fig_robots.canvas.flush_events()

"""

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.ion()  # Enable interactive mode

class RobotVisualizer_predicted:
    def __init__(self, bot_ids=None, figsize_robots=(12, 10)):
        # Create main robot position figure
        self.fig_robots = plt.figure(figsize=figsize_robots)
        self.ax_robots = self.fig_robots.add_subplot(111)
        
        # Initialize distance tracking
        self.current_time = 0
        
        # Set default bot_ids if none provided
        if bot_ids is None:
            bot_ids = [1, 2, 3, 4]
        
        # Define colors for each robot (ensure enough colors for all bots)
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
        self.robot_colors = {}
        for i, bot_id in enumerate(bot_ids):
            self.robot_colors[bot_id] = colors[i % len(colors)]
        
        # Initialize empty scatter plots for each robot
        self.scatters = {}
        self.predicted_scatters = {}
        self.prediction_lines = {}
        
        for bot_id in bot_ids:
            color = self.robot_colors[bot_id]
            self.scatters[bot_id] = self.ax_robots.scatter([], [], color=color, s=100, label=f'Robot {bot_id}')
            self.predicted_scatters[bot_id] = self.ax_robots.scatter([], [], color=color, marker='o', s=50, alpha=0.5)
            self.prediction_lines[bot_id] = None    

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
        phi_grad_arrow_legend = FancyArrowPatch((0, 0), (1, 0),
                                            color='blue',
                                            mutation_scale=15)
        prediction_line_legend = plt.Line2D([], [], color='gray', linestyle=':', label='Predicted Path')
        self.min_dist_line = plt.Line2D([], [], color='purple', linestyle='--', label='Min Distance')
        self.boundary_line = plt.Line2D([], [], color='black', linestyle='-', label='Boundary')
        
        # Create legend with all elements
        legend_elements = [
            *[self.scatters[bot_id] for bot_id in sorted(self.scatters.keys())],
            target_arrow_legend,
            current_arrow_legend,
            phi_grad_arrow_legend,
            prediction_line_legend,
            self.min_dist_line,
            self.boundary_line,
            self.obstacles_scatter
        ]
        legend_labels = [
            # Changed bot labels to robot labels
            *[f'Robot {bot_id}' for bot_id in sorted(self.scatters.keys())],
            'Target Heading',
            'Current Heading',
            'Phi Gradient',
            'Predicted Path',
            'Min Distance',
            'Boundary',
            'Obstacles'
        ]
        
        # Move legend to upper left corner
        self.ax_robots.legend(legend_elements, legend_labels, loc='upper left')
        
        # Setup robot position plot
        self.ax_robots.grid(True)
        self.ax_robots.set_xlabel('X Position (m)')
        self.ax_robots.set_ylabel('Y Position (m)')
        self.ax_robots.set_title("Robots' Current and Predicted Positions")
        
        # Show figure
        self.fig_robots.show()
    
    def update(self, bot_ids, robots_position, obstacles_position, robots_current_heading, 
          robots_current_speed, coords_predicted, min_distance_coordinates=None, boundary_parameters=None, phi_grad_arrows=None):
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
            
        # Remove previous prediction lines
        for bot_id in self.prediction_lines:
            if self.prediction_lines[bot_id] is not None:
                self.prediction_lines[bot_id].remove()
                self.prediction_lines[bot_id] = None
        
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
        
        # Position the text boxes at the right side, starting from the top
        y_start = 0.95  # Start higher up
        text_spacing = 0.15
        
        # Update each robot's data
        for i, bot_id in enumerate(bot_ids):
            # Ensure coordinates are scalar floats
            x = float(robots_position[0][i])
            y = float(robots_position[1][i])
            
            # Get predicted coordinates for this bot and ensure they're scalar floats
            pred_x = float(coords_predicted[2*i])
            pred_y = float(coords_predicted[2*i + 1])
            
            # Update scatter positions for current and predicted positions
            self.scatters[bot_id].set_offsets(np.array([[x, y]]))
            self.predicted_scatters[bot_id].set_offsets(np.array([[pred_x, pred_y]]))
            
            # Draw prediction path line - ensuring all values are scalar floats
            self.prediction_lines[bot_id] = self.ax_robots.plot([x, pred_x], [y, pred_y], 
                                                        color=self.robot_colors[bot_id],
                                                        linestyle=':', 
                                                        linewidth=1.5,
                                                        alpha=0.6)[0]
            
            # Add robot number labels
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
            
            # Calculate vector between current and predicted position
            target_x = pred_x - x
            target_y = pred_y - y
            
            # Calculate the distance between current and predicted position
            distance = np.sqrt(target_x**2 + target_y**2)
            
            # Scale the target heading arrow to 1.5 times the distance
            if distance > 0:
                scale_factor = 1.5
                scaled_x = target_x * scale_factor
                scaled_y = target_y * scale_factor
            else:
                # If distance is 0, use a small default value to avoid division by zero
                scaled_x = 0
                scaled_y = 0
            
            # Draw target heading arrow (red)
            target_arrow = self.ax_robots.arrow(x, y, 
                                        scaled_x, 
                                        scaled_y,
                                        head_width=1.0, 
                                        head_length=1.5,
                                        fc='red', 
                                        ec='red', 
                                        alpha=0.6)
            self.arrows.append(target_arrow)
            
            # Default values for gradient information
            grad_x = 0
            grad_y = 0
            grad_magnitude = 0
            
            # Draw phi gradient arrow if provided (using same scaling as target arrow)
            if phi_grad_arrows is not None and i < phi_grad_arrows.shape[1]:
                # Get the gradient components for this bot - row 0 is X direction, row 1 is Y direction
                grad_x = float(phi_grad_arrows[0, i])  # X-direction gradient
                grad_y = float(phi_grad_arrows[1, i])  # Y-direction gradient
                
                # Calculate magnitude
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Only draw if magnitude is not too small
                if grad_magnitude > 1e-6:
                    # Use same scale factor as target heading (1.5)
                    scale_factor = 1.5
                    scaled_grad_x = grad_x * scale_factor
                    scaled_grad_y = grad_y * scale_factor
                    
                    # Draw gradient arrow (blue for phi gradient)
                    grad_arrow = self.ax_robots.arrow(x, y, 
                                            scaled_grad_x, 
                                            scaled_grad_y,
                                            head_width=1.0, 
                                            head_length=1.5,
                                            fc='blue', 
                                            ec='blue', 
                                            alpha=0.7)
                    self.arrows.append(grad_arrow)
            
            # Current velocity vector - ensure scalar values
            current_speed = float(robots_current_speed[0][i])
            current_heading_rad = np.radians(float(robots_current_heading[0][i]))
            current_x = np.cos(np.radians(90) - current_heading_rad) * current_speed
            current_y = np.sin(np.radians(90) - current_heading_rad) * current_speed
            
            # Draw current heading arrow (green)
            current_arrow = self.ax_robots.arrow(x, y, 
                                        current_x * heading_scale, 
                                        current_y * heading_scale,
                                        head_width=1.0, 
                                        head_length=1.5,
                                        fc='green', 
                                        ec='green', 
                                        alpha=0.6)
            self.arrows.append(current_arrow)
            
            # Update text information including next waypoint distance and gradient info
            text = self.ax_robots.text(
                0.99, y_start - (i * text_spacing),  # Position at right edge
                f'$\\mathbf{{Robot\\ {bot_id}}}$\n'  # Bold robot name using LaTeX math syntax
                f'Current: X: {x:.1f}, Y: {y:.1f}\n'
                f'Predicted: X: {pred_x:.1f}, Y: {pred_y:.1f}\n'
                f'Next Waypoint: {distance:.2f} m\n'
                f'Current Speed: {current_speed:.2f} m/s\n'
                f'Current Heading: {robots_current_heading[0][i]:.1f}°\n'
                f'Gradient: X: {grad_x:.2f}, Y: {grad_y:.2f}, Mag: {grad_magnitude:.2f}',
                transform=self.ax_robots.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                ha='right',  # Right-align the text
                va='top'  # Align to top
            )
            self.texts.append(text)

        # Update minimum distance line
        if min_distance_coordinates is not None:
            dx = float(min_distance_coordinates[0][1]) - float(min_distance_coordinates[0][0])
            dy = float(min_distance_coordinates[1][1]) - float(min_distance_coordinates[1][0])
            distance = np.sqrt(dx**2 + dy**2)
            
            # Ensure scalar float values for plotting
            x0 = float(min_distance_coordinates[0][0])
            y0 = float(min_distance_coordinates[1][0])
            x1 = float(min_distance_coordinates[0][1])
            y1 = float(min_distance_coordinates[1][1])
            
            self.min_distance_line = self.ax_robots.plot(
                [x0, x1],
                [y0, y1],
                'purple', linestyle='--', linewidth=2, alpha=0.8
            )[0]
            
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            text = self.ax_robots.annotate(
                f'{distance:.1f} m',
                (mid_x, mid_y), xytext=(0, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                ha='center',
                va='bottom'
            )
            self.texts.append(text)
        
        # Update plot limits and refresh
        self.ax_robots.relim()
        self.ax_robots.autoscale_view()
        self.ax_robots.axis('equal')
        
        # Increment time
        self.current_time += 1
        
        # Redraw the plot
        self.fig_robots.canvas.draw()
        self.fig_robots.canvas.flush_events()
"""

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.ion()  # Enable interactive mode

class RobotVisualizer_predicted:
    def __init__(self, bot_ids=None, figsize_robots=(12, 10)):
        # Create main robot position figure
        self.fig_robots = plt.figure(figsize=figsize_robots)
        self.ax_robots = self.fig_robots.add_subplot(111)
        
        # Initialize distance tracking
        self.current_time = 0
        
        # Set default bot_ids if none provided
        if bot_ids is None:
            bot_ids = [1, 2, 3, 4]
        
        # Define colors for each robot (ensure enough colors for all bots)
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
        self.robot_colors = {}
        for i, bot_id in enumerate(bot_ids):
            self.robot_colors[bot_id] = colors[i % len(colors)]
        
        # Initialize empty scatter plots for each robot
        self.scatters = {}
        self.predicted_scatters = {}
        self.prediction_lines = {}
        
        for bot_id in bot_ids:
            color = self.robot_colors[bot_id]
            self.scatters[bot_id] = self.ax_robots.scatter([], [], color=color, s=100, label=f'Robot {bot_id}')
            self.predicted_scatters[bot_id] = self.ax_robots.scatter([], [], color=color, marker='o', s=50, alpha=0.5)
            self.prediction_lines[bot_id] = None 
            
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
        command_arrow_legend = FancyArrowPatch((0, 0), (1, 0),
                                            color='red',
                                            mutation_scale=15)
        current_arrow_legend = FancyArrowPatch((0, 0), (1, 0),
                                            color='green',
                                            mutation_scale=15)
        prediction_line_legend = plt.Line2D([], [], color='gray', linestyle=':', label='Predicted Path')
        self.min_dist_line = plt.Line2D([], [], color='purple', linestyle='--', label='Min Distance')
        self.boundary_line = plt.Line2D([], [], color='black', linestyle='-', label='Boundary')
        
        # Create legend with all elements
        legend_elements = [
            *[self.scatters[bot_id] for bot_id in sorted(self.scatters.keys())],
            command_arrow_legend,
            current_arrow_legend,
            prediction_line_legend,
            self.min_dist_line,
            self.boundary_line,
            self.obstacles_scatter
        ]
        legend_labels = [
            # Changed bot labels to robot labels
            *[f'Robot {bot_id}' for bot_id in sorted(self.scatters.keys())],
            'Target Heading',
            'Current Heading',
            'Predicted Path',
            'Min Distance',
            'Boundary',
            'Obstacles'
        ]
        
        # Move legend to upper left corner
        self.ax_robots.legend(legend_elements, legend_labels, loc='upper left')
        
        # Setup robot position plot
        self.ax_robots.grid(True)
        self.ax_robots.set_xlabel('X Position (m)')
        self.ax_robots.set_ylabel('Y Position (m)')
        self.ax_robots.set_title("Robots' Current and Predicted Positions")
        
        # Show figure
        self.fig_robots.show()
    
    def update(self, bot_ids, robots_position, obstacles_position, robots_current_heading, 
          robots_current_speed, coords_predicted, min_distance_coordinates=None, boundary_parameters=None):
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
            
        # Remove previous prediction lines
        for bot_id in self.prediction_lines:
            if self.prediction_lines[bot_id] is not None:
                self.prediction_lines[bot_id].remove()
                self.prediction_lines[bot_id] = None
        
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
        
        # Position the text boxes at the right side, starting from the top
        y_start = 0.95  # Start higher up
        text_spacing = 0.15
        
        # Update each robot's data
        for i, bot_id in enumerate(bot_ids):
            # Ensure coordinates are scalar floats
            x = float(robots_position[0][i])
            y = float(robots_position[1][i])
            
            # Get predicted coordinates for this bot and ensure they're scalar floats
            pred_x = float(coords_predicted[2*i])
            pred_y = float(coords_predicted[2*i + 1])
            
            # Update scatter positions for current and predicted positions
            self.scatters[bot_id].set_offsets(np.array([[x, y]]))
            self.predicted_scatters[bot_id].set_offsets(np.array([[pred_x, pred_y]]))
            
            # Draw prediction path line - ensuring all values are scalar floats
            self.prediction_lines[bot_id] = self.ax_robots.plot([x, pred_x], [y, pred_y], 
                                                        color=self.robot_colors[bot_id],
                                                        linestyle=':', 
                                                        linewidth=1.5,
                                                        alpha=0.6)[0]
            
            # Add robot number labels
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
            
            # Calculate vector between current and predicted position
            target_x = pred_x - x
            target_y = pred_y - y
            
            # Calculate the distance between current and predicted position
            distance = np.sqrt(target_x**2 + target_y**2)
            
            # Scale the target heading arrow to 1.5 times the distance
            if distance > 0:
                scale_factor = 1.5
                scaled_x = target_x * scale_factor
                scaled_y = target_y * scale_factor
            else:
                # If distance is 0, use a small default value to avoid division by zero
                scaled_x = 0
                scaled_y = 0
            
            # Draw target heading arrow (red)
            target_arrow = self.ax_robots.arrow(x, y, 
                                        scaled_x, 
                                        scaled_y,
                                        head_width=1.0, 
                                        head_length=1.5,
                                        fc='red', 
                                        ec='red', 
                                        alpha=0.6)
            self.arrows.append(target_arrow)
            
            # Current velocity vector - ensure scalar values
            current_speed = float(robots_current_speed[0][i])
            current_heading_rad = np.radians(float(robots_current_heading[0][i]))
            current_x = np.cos(np.radians(90) - current_heading_rad) * current_speed
            current_y = np.sin(np.radians(90) - current_heading_rad) * current_speed
            
            # Draw current heading arrow (green)
            current_arrow = self.ax_robots.arrow(x, y, 
                                        current_x * heading_scale, 
                                        current_y * heading_scale,
                                        head_width=1.0, 
                                        head_length=1.5,
                                        fc='green', 
                                        ec='green', 
                                        alpha=0.6)
            self.arrows.append(current_arrow)
            
            # Update text information including next waypoint distance - moved to right side
            text = self.ax_robots.text(
                0.99, y_start - (i * text_spacing),  # Position at right edge
                f'$\\mathbf{{Robot\\ {bot_id}}}$\n'  # Bold robot name using LaTeX math syntax
                f'Current: X: {x:.1f}, Y: {y:.1f}\n'
                f'Predicted: X: {pred_x:.1f}, Y: {pred_y:.1f}\n'
                f'Next Waypoint: {distance:.2f} m\n'  # Added waypoint distance
                f'Current Speed: {current_speed:.2f} m/s\n'
                f'Current Heading: {robots_current_heading[0][i]:.1f}°',
                transform=self.ax_robots.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                ha='right',  # Right-align the text
                va='top'  # Align to top
            )
            self.texts.append(text)

        # Update minimum distance line
        if min_distance_coordinates is not None:
            dx = float(min_distance_coordinates[0][1]) - float(min_distance_coordinates[0][0])
            dy = float(min_distance_coordinates[1][1]) - float(min_distance_coordinates[1][0])
            distance = np.sqrt(dx**2 + dy**2)
            
            # Ensure scalar float values for plotting
            x0 = float(min_distance_coordinates[0][0])
            y0 = float(min_distance_coordinates[1][0])
            x1 = float(min_distance_coordinates[0][1])
            y1 = float(min_distance_coordinates[1][1])
            
            self.min_distance_line = self.ax_robots.plot(
                [x0, x1],
                [y0, y1],
                'purple', linestyle='--', linewidth=2, alpha=0.8
            )[0]
            
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            text = self.ax_robots.annotate(
                f'{distance:.1f} m',
                (mid_x, mid_y), xytext=(0, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                ha='center',
                va='bottom'
            )
            self.texts.append(text)
        
        # Update plot limits and refresh
        self.ax_robots.relim()
        self.ax_robots.autoscale_view()
        self.ax_robots.axis('equal')
        
        # Increment time
        self.current_time += 1
        
        # Redraw the plot
        self.fig_robots.canvas.draw()
        self.fig_robots.canvas.flush_events()
"
"""