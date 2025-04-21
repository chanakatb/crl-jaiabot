import matplotlib.pyplot as plt
import numpy as np

class PhiPlotter:
    def __init__(self, figsize=(10, 6)):
        """Initialize the interactive plotter"""
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        
        # Store plot elements
        self.lines = []  # Store individual lines
        self.points = None  # Store scatter points
        self.text = None
        
        # Setup plot
        self.ax.grid(True)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('φ')
        self.ax.set_title('φ Value')
        
        # Initialize empty data
        self.phi_values = []
        self.time_points = []
        
        plt.show(block=False)
    
    def update(self, phi):
        """Update the plot with new phi value"""
        # Append new data
        self.phi_values.append(phi)
        self.time_points.append(len(self.time_points))
        
        # Clear previous elements
        for line in self.lines:
            line.remove()
        if self.points is not None:
            self.points.remove()
        if self.text is not None:
            self.text.remove()
        
        self.lines = []
        
        # Plot points
        self.points = self.ax.scatter(self.time_points, self.phi_values, 
                                    color='blue', s=50, zorder=5)
        
        # Draw lines between consecutive points
        for i in range(len(self.time_points)-1):
            line = self.ax.plot([self.time_points[i], self.time_points[i+1]],
                              [self.phi_values[i], self.phi_values[i+1]],
                              'b-', alpha=0.6, zorder=4)
            self.lines.extend(line)
        
        # Add current value text
        self.text = self.ax.text(
            0.95, 0.95,
            f'Current φ: {phi:.4f}',
            transform=self.ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
            ha='right',
            va='top'
        )
        
        # Update plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class GammaPlotter:
    def __init__(self, figsize=(10, 6)):
        """Initialize the interactive plotter"""
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        
        # Store plot elements
        self.lines = []  # Store individual lines
        self.points = None  # Store scatter points
        self.text = None
        
        # Setup plot
        self.ax.grid(True)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('γ')
        self.ax.set_title('γ Value')
        
        # Initialize empty data
        self.gamma_values = []
        self.time_points = []
        
        plt.show(block=False)
    
    def update(self, gamma):
        """Update the plot with new gamma value"""
        # Append new data
        self.gamma_values.append(gamma)
        self.time_points.append(len(self.time_points))
        
        # Clear previous elements
        for line in self.lines:
            line.remove()
        if self.points is not None:
            self.points.remove()
        if self.text is not None:
            self.text.remove()
        
        self.lines = []
        
        # Plot points
        self.points = self.ax.scatter(self.time_points, self.gamma_values,
                                    color='blue', s=50, zorder=5)
        
        # Draw lines between consecutive points
        for i in range(len(self.time_points)-1):
            line = self.ax.plot([self.time_points[i], self.time_points[i+1]],
                              [self.gamma_values[i], self.gamma_values[i+1]],
                              'b-', alpha=0.6, zorder=4)
            self.lines.extend(line)
        
        # Add current value text
        self.text = self.ax.text(
            0.95, 0.95,
            f'Current γ: {gamma:.4f}',
            transform=self.ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
            ha='right',
            va='top'
        )
        
        # Update plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class BeetaPlotter:
    def __init__(self, figsize=(10, 6)):
        """Initialize the interactive plotter"""
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        
        # Store plot elements
        self.lines = []  # Store individual lines
        self.points = None  # Store scatter points
        self.text = None
        
        # Setup plot
        self.ax.grid(True)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('β')
        self.ax.set_title('β Value')
        
        # Initialize empty data
        self.beeta_values = []
        self.time_points = []
        
        plt.show(block=False)
    
    def update(self, beeta):
        """Update the plot with new beeta value"""
        # Append new data
        self.beeta_values.append(beeta)
        self.time_points.append(len(self.time_points))
        
        # Clear previous elements
        for line in self.lines:
            line.remove()
        if self.points is not None:
            self.points.remove()
        if self.text is not None:
            self.text.remove()
        
        self.lines = []
        
        # Plot points
        self.points = self.ax.scatter(self.time_points, self.beeta_values,
                                    color='blue', s=50, zorder=5)
        
        # Draw lines between consecutive points
        for i in range(len(self.time_points)-1):
            line = self.ax.plot([self.time_points[i], self.time_points[i+1]],
                              [self.beeta_values[i], self.beeta_values[i+1]],
                              'b-', alpha=0.6, zorder=4)
            self.lines.extend(line)
        
        # Add current value text
        self.text = self.ax.text(
            0.95, 0.95,
            f'Current β: {beeta:.4f}',
            transform=self.ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
            ha='right',
            va='top'
        )
        
        # Update plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class MinDistancePlotter:
    def __init__(self, figsize=(10, 6)):
        """Initialize the interactive plotter for minimum distance visualization"""
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        
        # Store plot elements
        self.lines = []  # Store individual lines
        self.points = None  # Store scatter points
        self.text = None
        
        # Setup plot
        self.ax.grid(True)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Minimum Distance (m)')
        self.ax.set_title('Minimum Distance')
        
        # Initialize empty data
        self.min_distance_values = []
        self.time_points = []
        
        # Initialize without safety threshold
        
        plt.show(block=False)
    
    def update(self, min_distance):
        """Update the plot with new minimum distance value"""
        # Append new data
        self.min_distance_values.append(min_distance)
        self.time_points.append(len(self.time_points))
        
        # Clear previous elements
        for line in self.lines:
            line.remove()
        if self.points is not None:
            self.points.remove()
        if self.text is not None:
            self.text.remove()
        
        self.lines = []
        
        # Plot points
        self.points = self.ax.scatter(self.time_points, self.min_distance_values,
                                    color='blue', s=50, zorder=5)
        
        # Draw lines between consecutive points
        for i in range(len(self.time_points)-1):
            line = self.ax.plot([self.time_points[i], self.time_points[i+1]],
                              [self.min_distance_values[i], self.min_distance_values[i+1]],
                              'b-', alpha=0.6, zorder=4)
            self.lines.extend(line)
        
        # Add current value text
        self.text = self.ax.text(
            0.95, 0.95,
            f'Current Min Distance: {min_distance:.4f}m',
            transform=self.ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
            ha='right',
            va='top'
        )
        
        # Update plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Simple view adjustment
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

class MinDistanceFromBetaPlotter:
    def __init__(self, figsize=(10, 6)):
        """Initialize the interactive plotter for minimum distance visualization"""
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        
        # Store plot elements
        self.lines = []  # Store individual lines
        self.points = None  # Store scatter points
        self.text = None
        
        # Setup plot
        self.ax.grid(True)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Minimum Distance Approximation (m)')
        self.ax.set_title('Minimum Distance Approximation')
        
        # Initialize empty data
        self.min_distance_values = []
        self.time_points = []
        
        # Initialize without safety threshold
        
        plt.show(block=False)
    
    def update(self, min_distance):
        """Update the plot with new minimum distance value"""
        # Append new data
        self.min_distance_values.append(min_distance)
        self.time_points.append(len(self.time_points))
        
        # Clear previous elements
        for line in self.lines:
            line.remove()
        if self.points is not None:
            self.points.remove()
        if self.text is not None:
            self.text.remove()
        
        self.lines = []
        
        # Plot points
        self.points = self.ax.scatter(self.time_points, self.min_distance_values,
                                    color='blue', s=50, zorder=5)
        
        # Draw lines between consecutive points
        for i in range(len(self.time_points)-1):
            line = self.ax.plot([self.time_points[i], self.time_points[i+1]],
                              [self.min_distance_values[i], self.min_distance_values[i+1]],
                              'b-', alpha=0.6, zorder=4)
            self.lines.extend(line)
        
        # Add current value text
        self.text = self.ax.text(
            0.95, 0.95,
            f'Current Approx. Min Distance: {min_distance:.4f}m',
            transform=self.ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
            ha='right',
            va='top'
        )
        
        # Update plot limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Simple view adjustment
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

# # Example usage:
# plotter = PhiPlotter()

# # You can update the plot with new phi values:
# plotter.update(1.5)  # First value
# plotter.update(1.7)  # Second value
# plotter.update(1.2)  # Third value
