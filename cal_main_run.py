################################################################################################################################################
###################################################### Import Libraries
# Standard libraries
import os
import sys
import json
import time
import signal
import requests
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# User defined libraries
from input_parameters import * # import INPUT parameters
from cal_q_def import cal_q
from cal_obstacles_def import cal_obstacles
from cal_x_new_with_jaiabot_position_reading_def_ import cal_x_new_with_jaiabot_position_reading_
from cal_plot_robots_with_velocities_predicted_def import RobotVisualizer_predicted
from cal_get_coordinates_of_min_index_def import get_coordinates_of_min_index
from cal_nearest_points_to_bots_def import cal_nearest_points_to_bots
from cal_heading_and_speed_def import cal_heading_and_speed
from func_phi_plotter import PhiPlotter, GammaPlotter, BeetaPlotter, MinDistancePlotter, MinDistanceFromBetaPlotter
from func_plot_trajectories_handler_def import ProgramController, TrajectoryPlotter
from func_gamma import gamma_value_and_gradient
from func_min_distance_and_beta_with_smooth_approximation import min_distance_and_beta_with_smooth_approx
from func_phi import phi_and_its_gradient
from set_diving_def import set_diving
from set_diving_new_short_def import set_diving_new
""" For Jaiabot Simulator"""

from cal_jaiabot_lat_lon_position_def import cal_jaiabot_lat_lon_position
from cal_gps_to_local_coordinates_def import gps_to_local_coordinates
from cal_local_coordinates_to_gps_def import local_to_gps_coordinates
from cal_set_speed_and_direction_def import JaiabotController
from cal_set_way_point_def import JaiabotWaypointController
from func_bot_command_timer import BotCommandTimer
from func_format_bot_data_def import format_bot_data


################################################################################################################################################
###################################################### Main Algorithm's Parameters    

# Define parameters for Bots and Obstacle(s)
no_of_robots = len(bot_ids)
no_of_boundaries = no_of_robots
no_of_relative_distances_for_obstacles = no_of_robots * no_of_obstacles

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Bot command timer
command_timer = BotCommandTimer(interval=2.0)  # interval in second

# Relative position
desired_relative_positions_ = np.reshape(desired_relative_positions, 2*desired_relative_positions.shape[0])

# Take control
all_stop_resp = requests.post(url=API_ENDPOINT_ALL_STOP, headers=headers)
print("Stop all robots -> The pastebin URL is:%s"%all_stop_resp.text)

# Robots' intial position
robots_lat_lon, hub_lat_lon, robots_current_heading, robots_current_speed = cal_jaiabot_lat_lon_position(bot_ids=bot_ids,
                                                          api_endpoint=API_ENDPOINT_STATUS)
print('\n Robots lat and lon: \n', robots_lat_lon)
latitude_hub, longitude_hub = hub_lat_lon.flatten()
robots_position = gps_to_local_coordinates(robots_lat_lon, latitude_hub, longitude_hub, global_x_offset)

# Obstacles' position
obstacles_position = np.array([[-100], [-85]])
obstacles_position = cal_obstacles(obstacles_position)

# Nearest points on the boundary from each Bot
nearest_points_to_boundary_from_bots = cal_nearest_points_to_bots(np.hstack((robots_position, obstacles_position)), index_for_relative_positions, desired_relative_positions.shape[0], no_of_relative_distances_for_obstacles, origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse)

# Initialize positions
x = np.zeros((2, 2*no_of_robots + no_of_obstacles, 2))
x0_ = np.hstack((robots_position, obstacles_position, nearest_points_to_boundary_from_bots))
x[:, :, 0] = np.round(x0_, 4)

robots_lat_lon_predicted = np.zeros((2,4,2))

#===================================================================================
#   Initialize robots_position_new, x_predicted, and obstacles_position_new 
#   variables before getting into the main loop [HGT]
#---------------------------------------
robots_position_new = robots_position
obstacles_position_new = obstacles_position
# x_predicted_ = x
#====================================================================================

# Loop parameterrs
count = 0
running = True
mask = None

# First, create the plot
# visualizer = RobotVisualizer()
visualizer_predicted = RobotVisualizer_predicted(bot_ids=bot_ids)
phi_plotter = PhiPlotter()
gamma_plotter = GammaPlotter()
beeta_plotter = BeetaPlotter()
mindist_plotter = MinDistancePlotter()
mindist_from_beta_plotter = MinDistanceFromBetaPlotter()

# Time stamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create controller instance
controller = ProgramController()
jaiabot_controller = JaiabotController(hub_ip=HUB_IP)  # For controlling robots
jaiabotwaypoint_controller = JaiabotWaypointController(
        hub_ip=HUB_IP,
        api_endpoint_all_stop=API_ENDPOINT_ALL_STOP,
        api_endpoint_waypoint=API_ENDPOINT_WAYPOINT
    )

# Beta, Gamma, and Phi
beta_list = []
gamma_list = []
phi_list = []
min_dist_list = []
min_dist_from_beta_list = [] 

# Create a list to store timing results
timing_list = []
# Initialize a variable to store the start time
start_time = None

# Register the signal handler
def signal_handler(signum, frame):
    print("\nStopping execution...")    

    # # Save RobotVisualizer plots
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)    
 
    # Save Phi plot
    phi_plotter.fig.savefig(os.path.join('data', f'{timestamp}_phi.png'))
    # phi_plotter.fig.savefig(os.path.join('data', f'{timestamp}_phi.pdf'))

    # Save Gamma plot
    gamma_plotter.fig.savefig(os.path.join('data', f'{timestamp}_gamma.png'))
    # gamma_plotter.fig.savefig(os.path.join('data', f'{timestamp}_gamma.pdf'))
    
    # Save Beeta plot
    beeta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_beta.png'))
    # beeta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_beeta.pdf'))

    # Save Min Dist plot
    mindist_plotter.fig.savefig(os.path.join('data', f'{timestamp}_mindist.png'))
    # mindist_plotter.fig.savefig(os.path.join('data', f'{timestamp}_mindist.pdf'))

    # Save Min Dist plot
    mindist_from_beta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_min_dist_from_beta.png'))
    # mindist_approx_plotter.fig.savefig(os.path.join('data', f'{timestamp}_min_dist_from_beta.pdf'))

    # Create and save trajectory plot
    plotter = TrajectoryPlotter(
        x[:,:,:count+2], 
        count+1,
        origin_x_ellipse, 
        origin_y_ellipse,
        major_axis_ellipse, 
        minor_axis_ellipse,
        rotational_angle_ellipse,
        no_of_robots,    
        no_of_obstacles,
        timestamp
    )
    
    plotter.create_plot()
    plotter.save_plot()

    # Save arrays
    np.save(os.path.join('data', f'{timestamp}_x_array.npy'), x[:,:,:count+2])
    np.save(os.path.join('data', f'{timestamp}_robots_lat_lon_predicted.npy'), robots_lat_lon_predicted[:,:,:count+2])
    np.save(os.path.join('data', f'{timestamp}_hub_lat_lon.npy'), hub_lat_lon)

    np.save(os.path.join('data', f'{timestamp}_beta_values.npy'), np.array(beta_list))
    np.save(os.path.join('data', f'{timestamp}_gamma_values.npy'), np.array(gamma_list))
    np.save(os.path.join('data', f'{timestamp}_phi_values.npy'), np.array(phi_list))
    np.save(os.path.join('data', f'{timestamp}_phi_values.npy'), np.array(phi_list))
    np.save(os.path.join('data', f'{timestamp}_min_dist_values.npy'), np.array(min_dist_list))
    np.save(os.path.join('data', f'{timestamp}_min_dist_from_beta_values.npy'), np.array(min_dist_from_beta_list))
    np.save(os.path.join('data', f'{timestamp}_timing_values.npy'), np.array(timing_list))
    
    print(f"\nAll plots saved with timestamp: {timestamp}")
    print(f"Files saved in 'data' directory:")
    print(f"- {timestamp}_robots_position.png/pdf")
    print(f"- {timestamp}_distance.png/pdf")
    print(f"- {timestamp}_phi.png/pdf")
    print(f"- {timestamp}_gamma.png/pdf")
    print(f"- {timestamp}_beta.png/pdf")
    print(f"- {timestamp}_mindist.png/pdf") 
    print(f"- {timestamp}_x_array.npy")   
    print(f"- {timestamp}_robots_lat_lon_predicted.npy")
    print(f"- {timestamp}_hub_lat_lon.npy")

    print(f"- {timestamp}_beta_values.npy")
    print(f"- {timestamp}_gamma_values.npy") 
    print(f"- {timestamp}_phi_values.npy")
    print(f"- {timestamp}_min_dist_values.npy")
    print(f"- {timestamp}_min_dist_from_beta_values.npy")
    print(f"- {timestamp}_timing_values.npy")

    # Stop the main loop
    print("==== controller.running = False ====")
    controller.running = False    
    
    # System exit
    raise SystemExit


# Register the signal handler directly
signal.signal(signal.SIGINT, signal_handler)

# Read Bot data
# command_velocity = np.zeros((2,len(bot_ids)))
bot_data = format_bot_data(bot_ids, robots_current_heading, robots_current_speed)

# Last command
last_command = False
################################################################################################################################################
###################################################### Main Algorithm     

while controller.running:     
    if count == 0:
        # Start the stopwatch on the first iteration
        start_time = time.time()
    start_time_main_loop = time.time()

    print(f"\n****************** Iteration ****************** : {count}")
    print(f"x[:,:,{count}]: \n", x[:,:,count], "\n" )

    # Grow arrays if needed
    if count >= x.shape[2] - 1:
        x = np.concatenate([x, np.zeros_like(x)], axis=2)
        robots_lat_lon_predicted = np.concatenate([robots_lat_lon_predicted, np.zeros_like(robots_lat_lon_predicted)], axis=2)

    # Evaluate relative positions in SCALED COORDINATE
    q = cal_q(x[:,:,count], index_for_relative_positions, desired_relative_positions.shape[0], 
             no_of_relative_distances_for_obstacles, scaling_factor, origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse)
    q_reshaped = np.reshape(q, (2, index_for_relative_positions.shape[0]), order='F') 
    relativeDiffError = np.round((1/scaling_factor) * (np.reshape(q[:(2*desired_relative_positions.shape[0]), :], (2, desired_relative_positions.shape[0]), order='F') + desired_relative_positions.T), 4)
    print("Relative Position Error:\n", relativeDiffError)    

    # Position of Bots
    coords = [x[0,0,count], x[1,0,count], # x0, y0 -> Bot 1
              x[0,1,count], x[1,1,count], # x1, y1 -> Bot 2
              x[0,2,count], x[1,2,count], # x2, y2 -> Bot 3
              x[0,3,count], x[1,3,count], # x3, y3 -> Bot 4  
              x[0,4,count], x[1,4,count], # x4, y4 -> Obstacle  
              x[0,5,count], x[1,5,count], # x5, y5 -> Boundary 1  
              x[0,6,count], x[1,6,count], # x6, y6 -> Boundary 2   
              x[0,7,count], x[1,7,count], # x7, y7 -> Boundary 3 
              x[0,8,count], x[1,8,count], # x8, y8 -> Boundary 4 
              ]    
        
    # Horizon loop
    print('\n------ Prediction Horizon ------\n')
    
    # ----- use ORIGINAL COORDINATE -----
    coords_predicted = np.array(coords)
    # ----- use SCALED COORDINATE -----    
    coords_predicted_scaled = scaling_factor * np.array(coords)
    
    # ----- use ORIGINAL COORDINATE -----
    prediction_waypoint = robots_position_new
    
    # Start timing the prediction horizon loop
    prediction_horizon_start_time = time.time()

    for k in range(prediction_horizon):
        # Evaluate beta fuction        
        # ----- use SCALED COORDINATE -----    
        min_dist_from_beta, min_idx_from_beta, beta_value, beta_gradient, all_distances  = min_distance_and_beta_with_smooth_approx(
            B, rho, mu_nf, a_nf, r_nf, *coords_predicted_scaled)    
        # print(f"beta_value [{k}/{prediction_horizon}]:\n", beta_value)
        # print(f"beta_gradient [{k}/{prediction_horizon}]:\n", beta_gradient)
       
        # Evaluate gamma and its derivatives 
        # ----- use SCALED COORDINATE -----           
        gamma_value, gamma_gradient, _ = gamma_value_and_gradient(B_gamma, B, desired_relative_positions, coords_predicted_scaled)
        # print(f"\ngamma_value [{k}/{prediction_horizon}]:\n", gamma_value)
        # print(f"gamma_gradient [{k}/{prediction_horizon}]:\n", gamma_gradient)
        
        # Evaluate phi and its derivatives 
        # ----- use SCALED COORDINATE -----             
        phi_val, phi_grad = phi_and_its_gradient(gamma_value, beta_value, gamma_gradient, beta_gradient, k_Beta_nf) 
        # print(f"\nphi_val [{k}/{prediction_horizon}]:\n", phi_val)
        # print(f"phi_grad [{k}/{prediction_horizon}]:\n", phi_grad)

        # Velocity of Bots
        output_velocity = np.reshape(-1*phi_grad[:2*no_of_robots], (2, no_of_robots ), order='F' )
        # print(f"output_velocity [{k}/{prediction_horizon}]:\n", output_velocity)    

        # Calculate the phi gradient for visualization
        phi_grad_visualization = np.reshape(-1*phi_grad[:2*no_of_robots], (2, no_of_robots), order='F')
        
        # Save initial values in prediction horizon
        if k==0:
            min_dist_from_beta_0 = min_dist_from_beta
            beta_value_0 = beta_value
            gamma_value_0 = gamma_value
            phi_val_0 = phi_val
            print(f"phi_val[{k+1}]: {phi_val} \n")
                          
        # Commanded velocity normalization
        command_speed = np.zeros(no_of_robots)
        command_velocity = np.zeros((2,no_of_robots))
        epsilon = 1e-10  # Small value to prevent division by zero
        
        # Potential field reference normalization
        # Convert to standard float types
        output_velocity = np.array(output_velocity, dtype=np.float64)
        for i in range(0, no_of_robots):
            norm = np.linalg.norm(output_velocity[:,i])
            command_speed[i] = np.arctan(300 * norm)  

            # Only normalize if the norm is not too close to zero
            if norm > epsilon:
                command_velocity[:,i] = command_speed[i] * output_velocity[:,i] / norm
            else:
                command_velocity[:,i] = 0  # Set to zero if the velocity is negligible
        # print(f"Command velocity [{k}/{prediction_horizon}]: \n", command_velocity)       
                         
        # if phi_val_0 < 1e-1:
        if np.linalg.norm(command_velocity) < 1.0 and np.all(np.linalg.norm(relativeDiffError, axis=0) < 10):
            print("--- Switching to the final waypoint command ---")

            # Calculate the norm of each column
            col_norms = np.linalg.norm(command_velocity, axis=0)

            # Divide each column by its norm to get unit vectors
            unit_command_velocity = command_velocity / col_norms           

            # Remaining position error when the bots stop moving
            # ----- use ORIGINAL COORDINATE -----
            last_position_of_bots = x[:,:4,count] + (unit_command_velocity* final_speed )* final_moving_time
            
            # ----- use ORIGINAL COORDINATE -----
            robots_lat_lon_predicted[:,:,count] = local_to_gps_coordinates(last_position_of_bots, latitude_hub, longitude_hub, global_x_offset=global_x_offset)
            robots_lat_lon_predicted[:,:,count] = np.round(robots_lat_lon_predicted[:,:,count], 6)

            # Send waypoint commands to all bots
            print('\n\n-----===== Sending waypoint commands to all bots (last)====-----\n')
            print(f'Robots predicted lat and lon [:,:,{count}]: \n', robots_lat_lon_predicted[:,:,count])     
            jaiabotwaypoint_controller.control_bots_(robots_lat_lon_predicted[:,:,count], bot_ids)  
            
            # Get updated bot data
            bot_data_predicted = jaiabotwaypoint_controller.get_bot_data_(robots_lat_lon_predicted[:,:,count], bot_ids)
            
            print("\n" * 3)
            print("+++++++++++++" * 2, "--- --- The formation is achived. --- ---", "+++++++++++++" * 2)
                        
            # Wait till the bots stop
            time.sleep(1.5*final_moving_time)

            # New positions  
            robots_lat_lon, hub_lat_lon, robots_current_heading, robots_current_speed = cal_jaiabot_lat_lon_position(bot_ids=bot_ids,
                                                                    api_endpoint=API_ENDPOINT_STATUS)
            print('Robots lat and lon: \n', robots_lat_lon)
            # ----- use ORIGINAL COORDINATE -----
            robots_position_new = gps_to_local_coordinates(robots_lat_lon, latitude_hub, longitude_hub, global_x_offset)

            # ----- use ORIGINAL COORDINATE -----
            obstacles_position_new = x[:,no_of_robots:(no_of_robots+no_of_obstacles),count] # obstacles positions remain unchanged.
            print("\nRobots position:\n", robots_position_new)   
            # print("Obstacles_position_new:\n", obstacles_position_new)       
                        
            # Dive command
            print("\n ------ Dive Commands ------")
            # # Method A: Sequential commands
            # for ID, max_depth, depth_interval, hold_time, drift_time in zip(bot_ids, max_depths, depth_intervals, hold_times,  drift_times):
            #     set_diving(ID, max_depth, depth_interval, hold_time, drift_time, API_ENDPOINT_RC, API_ENDPOINT_STATUS, API_ENDPOINT_ALL_STOP)

            # Method B: Parallel commands
            set_diving_new(bot_ids, max_depths, depth_intervals, hold_times, drift_times, API_ENDPOINT_RC, API_ENDPOINT_STATUS, API_ENDPOINT_ALL_STOP)      
            
            # Stop the main loop
            controller.running = False
            last_command = True
            break
            

        # Predicted waypoint update
        # ----- use ORIGINAL COORDINATE -----
        prediction_waypoint = prediction_waypoint + command_velocity * delta_t  
        # prediction_waypoint = robots_position_new

        # Obstacle upate
        # ----- use ORIGINAL COORDINATE -----
        obstacles_position_new = x[:,no_of_robots:(no_of_robots+no_of_obstacles),count] # obstacles positions remain unchanged.
        # ----- use ORIGINAL COORDINATE -----
        x_predicted  = cal_x_new_with_jaiabot_position_reading_(x0_, prediction_waypoint, no_of_robots, no_of_obstacles)
        x_predicted_ = np.round(x_predicted, 4)
        # ----- use ORIGINAL COORDINATE -----
        nearest_points_to_boundary_from_waypoint =  cal_nearest_points_to_bots(x_predicted_, index_for_relative_positions, desired_relative_positions.shape[0], no_of_relative_distances_for_obstacles, origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse)
        x_predicted_[:, -nearest_points_to_boundary_from_waypoint.shape[1]:] = nearest_points_to_boundary_from_waypoint
        x_predicted_ = np.round(x_predicted_, 4)
        
        # ----- use ORIGINAL COORDINATE -----
        coords_predicted = [x_predicted_[0,0], x_predicted_[1,0], # x0, y0 -> waypoint for Bot 1
                            x_predicted_[0,1], x_predicted_[1,1], # x1, y1 -> waypoint for Bot 2
                            x_predicted_[0,2], x_predicted_[1,2], # x2, y2 -> waypoint for Bot 3
                            x_predicted_[0,3], x_predicted_[1,3], # x3, y3 -> waypoint for Bot 4  
                            x_predicted_[0,4], x_predicted_[1,4], # x4, y4 -> Obstacle   
                            x_predicted_[0,5], x_predicted_[1,5], # x5, y5 -> predicted Boundary 1  
                            x_predicted_[0,6], x_predicted_[1,6], # x6, y6 -> predicted Boundary 2  
                            x_predicted_[0,7], x_predicted_[1,7], # x7, y7 -> predicted Boundary 3 
                            x_predicted_[0,8], x_predicted_[1,8], # x8, y8 -> predicted Boundary 4 
                            ] 
        # ----- use ORIGINAL COORDINATE -----
        coords_predicted = np.array(np.round(coords_predicted, 4))
        # ----- use SCALED COORDINATE -----  
        coords_predicted_scaled = scaling_factor * np.array(coords_predicted)
        # print(f"coords_predicted [{k}/{prediction_horizon}]: \n", np.array(coords_predicted[:8]).reshape(2, -1)) 
        
    # End timing the prediction horizon loop
    prediction_horizon_end_time = time.time()
    prediction_horizon_elapsed_time = prediction_horizon_end_time - prediction_horizon_start_time
                                    

    # Update visualizer_predicted     
    # ----- use ORIGINAL COORDINATE -----
    min_distance_coordinates, minimum_index, minimum_distance = get_coordinates_of_min_index(x[:,:,count], index_for_relative_positions)
    visualizer_predicted.update(
    bot_ids, 
    robots_position_new, 
    obstacles_position_new, 
    robots_current_heading, 
    robots_current_speed, 
    coords_predicted[:8], 
    min_distance_coordinates, 
    boundary_parameters,
    phi_grad_visualization
    )
    
    # Send waypoint commands to Bots
    if command_timer.can_send_command() and not last_command:
        # Slice the first 8 elements (x, y for the 4 bots)
        # ----- use ORIGINAL COORDINATE -----
        coords_predicted_bots = np.array(coords_predicted[:8])
        # Separate x and y coordinates
        # ----- use ORIGINAL COORDINATE -----
        coords_predicted_bots_reshaped = np.array([
            coords_predicted_bots[::2],  # x-coordinates
            coords_predicted_bots[1::2]  # y-coordinates
            ])
        # ----- use ORIGINAL COORDINATE -----
        robots_lat_lon_predicted[:,:,count] = local_to_gps_coordinates(coords_predicted_bots_reshaped, latitude_hub, longitude_hub, global_x_offset=global_x_offset)
        robots_lat_lon_predicted[:,:,count]  = np.round(robots_lat_lon_predicted[:,:,count] , 6)

        # Send waypoint commands to all bots        
        print(f'Robots predicted lat and lon [:,:,{count}] : \n', robots_lat_lon_predicted[:,:,count] )  
        print('\n\n-----===== Started sending waypoint commands to all bots (main)====-----\n')   
        jaiabotwaypoint_controller.control_bots_(robots_lat_lon_predicted[:,:,count], bot_ids)  
        print('\n\n-----===== Completed sending waypoint commands to all bots (main)====-----\n')
        
        # Get updated bot data
        bot_data_predicted = jaiabotwaypoint_controller.get_bot_data_(robots_lat_lon_predicted[:,:,count], bot_ids)       
    #==================================================================================
    
    # Read Bot data
    bot_data = format_bot_data(bot_ids, robots_current_heading, robots_current_speed)
    if not last_command:
        for bot in bot_data:
            print(f"\nBot {bot['bot_id']}:")
            print(f"Current Speed: {bot['speed']:.2f} m/s")
            print(f"Current Heading: {bot['heading']:.2f}°")
    
    # New positions  
    robots_lat_lon, hub_lat_lon, robots_current_heading, robots_current_speed = cal_jaiabot_lat_lon_position(bot_ids=bot_ids,
                                                              api_endpoint=API_ENDPOINT_STATUS)
    print('\nRobots lat and lon: \n', robots_lat_lon)

    # ----- use ORIGINAL COORDINATE -----
    robots_position_new = gps_to_local_coordinates(robots_lat_lon, latitude_hub, longitude_hub, global_x_offset)

    # ----- use ORIGINAL COORDINATE -----
    obstacles_position_new = x[:,no_of_robots:(no_of_robots+no_of_obstacles),count] # obstacles positions remain unchanged.
    # print("\nRobots position:\n", robots_position_new)   
    # print("Obstacles_position_new:\n", obstacles_position_new)        

    # Store new positions
    # ----- use ORIGINAL COORDINATE -----
    x_new = cal_x_new_with_jaiabot_position_reading_(x0_, robots_position_new, no_of_robots, no_of_obstacles)
    x_new_ = np.round(x_new, 4) 
    nearest_points_to_boundary_from_bots = cal_nearest_points_to_bots(x[:,:,count], index_for_relative_positions, desired_relative_positions.shape[0], no_of_relative_distances_for_obstacles, origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse)
    x_new_[:, -nearest_points_to_boundary_from_bots.shape[1]:] = nearest_points_to_boundary_from_bots    
    x_new_ = np.round(x_new_, 4)
    print('Positions (Robot, Obstacle, and Boudary closest to each Bot): \n', x_new_)
    x[:, :, count + 1] = x_new_      
        
    # Append beta value, gamma value, and phi value        
    gamma_list.append(gamma_value_0)
    beta_list.append(beta_value_0)
    phi_list.append(phi_val_0) 
    min_dist_list.append(minimum_distance) 
    min_dist_from_beta_list.append(min_dist_from_beta_0*(1/scaling_factor))
    
    main_loop_time = np.round(time.time() - start_time_main_loop, 4)
    elapsed_time = np.round(time.time() - start_time, 4)
    timing_list.append(elapsed_time)

    # Print the elapsed time
    print(f"\nPrediction Horizon loop time: {prediction_horizon_elapsed_time:.4f} seconds")
    print(f"Main loop time: {main_loop_time:.2f} seconds")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
    # Upate plot    
    gamma_plotter.update(gamma_value_0)
    beeta_plotter.update(beta_value_0)
    phi_plotter.update(phi_val_0)    
    mindist_plotter.update(minimum_distance)
    mindist_from_beta_plotter.update(min_dist_from_beta_0*(1/scaling_factor))

    if last_command == True:       
            # Save Phi plot
            phi_plotter.fig.savefig(os.path.join('data', f'{timestamp}_phi.png'))
            # phi_plotter.fig.savefig(os.path.join('data', f'{timestamp}_phi.pdf'))
            
            # Save Gamma plot
            gamma_plotter.fig.savefig(os.path.join('data', f'{timestamp}_gamma.png'))
            # gamma_plotter.fig.savefig(os.path.join('data', f'{timestamp}_gamma.pdf'))
            
            # Save Beeta plot
            beeta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_beta.png'))
            # beeta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_beta.pdf'))

            # Save Min Dist plot
            mindist_plotter.fig.savefig(os.path.join('data', f'{timestamp}_mindist.png'))
            # mindist_plotter.fig.savefig(os.path.join('data', f'{timestamp}_mindist.pdf'))   

            # Save Min Dist plot
            mindist_from_beta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_min_dist_from_beta.png'))
            # mindist_from_beta_plotter.fig.savefig(os.path.join('data', f'{timestamp}_min_dist_from_beta.pdf'))         

            plotter = TrajectoryPlotter(
                x[:,:,:count+2], 
                count+1,
                origin_x_ellipse, 
                origin_y_ellipse,
                major_axis_ellipse, 
                minor_axis_ellipse,
                rotational_angle_ellipse,
                no_of_robots,    
                no_of_obstacles,
                timestamp
            )

            # Create and save plot
            plotter.create_plot()
            plotter.save_plot()       

            # Save arrays
            np.save(os.path.join('data', f'{timestamp}_x_array.npy'), x[:,:,:count+2])
            np.save(os.path.join('data', f'{timestamp}_robots_lat_lon_predicted.npy'), robots_lat_lon_predicted[:,:,:count+2]) 
            np.save(os.path.join('data', f'{timestamp}_hub_lat_lon.npy'), hub_lat_lon)

            np.save(os.path.join('data', f'{timestamp}_beta_values.npy'), np.array(beta_list))
            np.save(os.path.join('data', f'{timestamp}_gamma_values.npy'), np.array(gamma_list))
            np.save(os.path.join('data', f'{timestamp}_phi_values.npy'), np.array(phi_list))
            np.save(os.path.join('data', f'{timestamp}_min_dist_values.npy'), np.array(min_dist_list))
            np.save(os.path.join('data', f'{timestamp}_min_dist_from_beta_values.npy'), np.array(min_dist_from_beta_list))
            np.save(os.path.join('data', f'{timestamp}_timing_values.npy'), np.array(timing_list))
    

            print(f"\nAll plots saved with timestamp: {timestamp}")
            print(f"Files saved in 'data' directory:")
            print(f"- {timestamp}_robots_position.png/pdf")
            print(f"- {timestamp}_distance.png/pdf")
            print(f"- {timestamp}_phi.png/pdf")
            print(f"- {timestamp}_gamma.png/pdf")
            print(f"- {timestamp}_beta.png/pdf")
            print(f"- {timestamp}_mindist.png/pdf") 
            print(f"- {timestamp}_x_array.npy")   
            print(f"- {timestamp}_robots_lat_lon_predicted.npy")
            print(f"- {timestamp}_hub_lat_lon.npy")

            print(f"- {timestamp}_beta_values.npy")
            print(f"- {timestamp}_gamma_values.npy") 
            print(f"- {timestamp}_phi_values.npy")
            print(f"- {timestamp}_min_dist_values.npy")
            print(f"- {timestamp}_min_dist_from_beta_values.npy")
            print(f"- {timestamp}_timing_values.npy")

            print("==== last_command == True ====")
            # print("==== END ====")
    
    
    # Count
    count += 1
    time.sleep(0.2) 

