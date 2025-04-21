# importing the requests library
import requests
import numpy as np
 
# BOT IDs and NO of OBSTACLES
bot_ids = [1,2,3,4]  # change [2, 6, 7, 8]  
no_of_obstacles = 1

# NAVIGATIONAL FUNCTION PARAMETERS
a_nf = 2.7182818
mu_nf = 2
r_nf = -1
k_Beta_nf = 1
scaling_factor = 0.005
rho = 1000 # 10

# PREDICTION HORIZON
delta_t = 0.15 # 0.30 # Time step for the numerical integration of the position in seconds
prediction_horizon = 150 # 40 # No of steps that integrate the gradient of PHI

# FINAL COMMAND AFTER REACHING THE FORMATION
final_speed = 0  # final speed in m/s # 0.5
final_moving_time = 0 # # final moving time in seconds # 6

# DIVING PARAMETERS
max_depths = [0.5, 0.9, 1.7, 1.7] # 0.828 x (depth - 0.918)
depth_intervals = [0.5, 0.9, 1.7, 1.7]
hold_times = [60, 60, 60, 60]
drift_times = [0, 0, 0, 0]

# GLOBAL COORDINATE OFFSET FORM EAST
global_x_offset = 0 # in degree

# ANGLE TO THE FORMATION LINE FORMED FROM EAST IN COUNTER CLOCKWISE DIRECTION
theta = np.radians(45) # in degree

# ROTATIONAL MATRIX
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
    ])

# TARGET RELATIVE POSITIONS BETWEEN ROBOTS
desired_relative_positions = scaling_factor * np.dot(np.array([[10, 0.0], [10, 0.0], [-10, 0], [20, 0], [-20, 0], [30, 0]]), rotation_matrix.T) # for line formation
# desired_relative_positions = scaling_factor * np.dot(np.array([[20, 0.0], [-10, -10], [0, -20], [-10, 10], [-10, 10], [10, 10]]), rotation_matrix.T) # for diamond formation
print(desired_relative_positions)
# API ENDPOINTS
HUB_IP = "localhost:40001"  # change 10.23.10.11 or localhost:40001
API_ENDPOINT_ALL_STOP = f"http://{HUB_IP}/jaia/all-stop"
API_ENDPOINT_MANUAL = f"http://{HUB_IP}/jaia/ep-command"
API_ENDPOINT_STATUS = f"http://{HUB_IP}/jaia/status"
API_ENDPOINT_RC = f"http://{HUB_IP}/jaia/command" # when running on Hub 11
API_ENDPOINT_WAYPOINT = f"http://{HUB_IP}/jaia/single-waypoint-mission"
headers = {'clientid': 'backseat-control', 'Content-Type': 'application/json; charset=utf-8'}

# MATRIX B
B = np.array([[ 1, -1,  0,  0,  0,  0,  0,  0,  0],
              [ 0,  1, -1,  0,  0,  0,  0,  0,  0],
              [ 0,  0, -1,  1,  0,  0,  0,  0,  0],
              [ 0,  1,  0, -1,  0,  0,  0,  0,  0],
              [-1,  0,  1,  0,  0,  0,  0,  0,  0],
              [ 1,  0,  0, -1,  0,  0,  0,  0,  0],
              [ 1,  0,  0,  0, -1,  0,  0,  0,  0],
              [ 0,  1,  0,  0, -1,  0,  0,  0,  0],
              [ 0,  0,  1,  0, -1,  0,  0,  0,  0],
              [ 0,  0,  0,  1, -1,  0,  0,  0,  0],
              [ 1,  0,  0,  0,  0, -1,  0,  0,  0],
              [ 0,  1,  0,  0,  0,  0, -1,  0,  0],
              [ 0,  0,  1,  0,  0,  0,  0, -1,  0],
              [ 0,  0,  0,  1,  0,  0,  0,  0, -1]])
B_gamma = B[:6, :4]
# RELATIVE POSITION INDEX
index_for_relative_positions = np.array([[0, 1], [1, 2], [3, 2], [1, 3], [2, 0], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8]])


# ELLIPSE PARAMETERS (BOUNDARY)
origin_x_ellipse = -95
origin_y_ellipse = -75
major_axis_ellipse = 50 # semi-major axis 70 60 50
minor_axis_ellipse= 30  # semi-minor axis 50 40 30
rotational_angle_ellipse = np.radians(45)  # rotation angle in degrees

boundary_parameters = [origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse]

# SAFETY FACTOR FOR THE BOUNDARY
scaling_factor_for_boundary = 1