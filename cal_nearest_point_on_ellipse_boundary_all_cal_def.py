import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def point_in_ellipse_(x, y, center_x, center_y, a, b, phi):
    # Convert Cartesian coordinates to polar coordinates
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)

    # Calculate expected radius using the polar equation of the ellipse
    expected_r = (a * b) / np.sqrt((b * np.cos(theta - phi))**2 + (a * np.sin(theta - phi))**2)
    
    
    # Compare the radius to the expected radius
    if np.isclose(r, expected_r):
        return "On"
    elif r < expected_r:
        return "In"
    else:
        return "Out"    


def find_nearest_point_all_cal(ellipse_params, point_of_interested, searching_angle):
    # Define parameters
    center_x, center_y, a, b, phi = ellipse_params
    x0, y0 = point_of_interested

    # Define the parametric equations for x(t) and y(t)
    safety_factor =1 # This prevents the nearest points located on the boundary of the ellipse, promising the points belongs to one of the ellipse.
    a = safety_factor*a
    b = safety_factor*b
    def x(t):
        x = center_x + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        return x

    def y(t):
        y = center_y + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
        return y

    # Define the function to calculate the distance between two points
    def distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Define the interval
    interval = 1e-3
    # Define the intial distance
    min_distance = a
    # Define the minimum t
    t_min = 0

    # Create an array of values from 0 to 2*pi with the specified interval    
    for t in searching_angle:
        dist = distance(x(t), y(t), x0, y0)
        if dist < min_distance:
            min_distance = dist
            t_min = t     
    
    # The nearest point
    nearest_point = [x(t_min), y(t_min)]

    return nearest_point, min_distance

def cal_nearest_point_to_boundary(point_x, point_y, center_x, center_y, a, b, phi):
    # Determine the position of the point relative to the ellipses
    found = False
    ellipse_no = []

    # Check each ellipse
    for i, (center_x, center_y, a, b, phi) in enumerate([(center_x, center_y, a, b, phi)]):
        position = point_in_ellipse_(point_x, point_y, center_x, center_y, a, b, phi)
        if position in ["On", "In"]:
            # print(f"The point ({point_x}, {point_y}) is in ellipse {3-i}")
            # ellipse_no = 3-i
            ellipse_no.append(i)
            found = True
            # break

    if not found:
        ellipse_no = "not in any of the ellipses"
        # print(f"The point ({point_x}, {point_y}) is not in any of the ellipses")

    # Define the interval
    interval = 1e-2
    if ellipse_no[0] == 0:  
        searching_angle = np.arange(0, 2*np.pi, interval) 
        ellipse_params = [center_x, center_y, a, b, phi]   
    else:
        print("Error: The robot is out of the boundary!")
        
    nearest_point, min_distance = find_nearest_point_all_cal(ellipse_params, [point_x, point_y], searching_angle)
    nearest_point = np.array(nearest_point)
    # print(f"Ellipse NO:{ellipse_no[0]}, Nearest Point: {nearest_point}\n")
    nearest_point = np.reshape(nearest_point, (2,1))


    return nearest_point, min_distance