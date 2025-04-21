import math
import requests
import json
import time 
import numpy as np
from math import atan2, degrees
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt

def initialize_control(hub_ip):
    """
    Initialize API endpoints and take control of the bots.
    
    Args:
        hub_ip (str): IP address of the hub
        
    Returns:
        dict: API endpoints
        dict: Request headers
    """
    # Define API endpoints
    api_endpoints = {
        'all_stop': f"http://{hub_ip}/jaia/all-stop",
        'manual': f"http://{hub_ip}/jaia/ep-command"
    }
    
    # Define request headers
    headers = {
        'clientid': 'backseat-control',
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    # Send all stop to take control
    all_stop_resp = requests.post(url=api_endpoints['all_stop'], headers=headers)
    print(f"The pastebin URL is: {all_stop_resp.text}")
    
    return api_endpoints, headers

class JaiabotController:
    def __init__(self, hub_ip="10.23.10.11"):
        self.API_ENDPOINT_ALL_STOP = f"http://{hub_ip}/jaia/all-stop"
        self.API_ENDPOINT_MANUAL = f"http://{hub_ip}/jaia/ep-command"
        
        self.headers = {
            'clientid': 'backseat-control',
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        self.speed_params = {
            'max_speed': 2.0,
            'min_speed': 0,
            'max_speed_org': 2.0,
            'max_safety_speed': 2.0,
            'min_safety_speed': 0.0,
            'min_operational_speed': 0 # 0.2
        }
        # self.speed_params = {
        #     'max_speed': 3.0,
        #     'min_speed': 0,
        #     'max_speed_org': 3.0,
        #     'max_safety_speed': 3.0,
        #     'min_safety_speed': 0.0,
        #     'min_operational_speed': 0.2
        # }
    
    def take_control(self):
        response = requests.post(
            url=self.API_ENDPOINT_ALL_STOP,
            headers=self.headers
        )
        print(f"All Stop Response: {response.text}")
        return response.ok
    
    @staticmethod
    def clamp(value, min_val, max_val):
        return max(min_val, min(max_val, value))
    
    def map_speed(self, speed):
        clamped_speed = self.clamp(speed, 0, self.speed_params['max_speed_org'])
        normalized_speed = clamped_speed / self.speed_params['max_speed_org']
        mapped_speed = (self.speed_params['min_operational_speed'] + 
                       normalized_speed * (self.speed_params['max_speed'] - 
                                        self.speed_params['min_speed']))
        limited_speed = self.clamp(mapped_speed,
                                 self.speed_params['min_safety_speed'],
                                 self.speed_params['max_safety_speed'])
        return limited_speed
    
    @staticmethod
    def alpha_to_theta(alpha, global_x_offset):
        """
        Convert angle from global reference frame (alpha) to compass heading (theta).
        Global reference frame is right-hand oriented.
        
        Args:
            alpha (float): Angle in radians from global frame (atan2(vy, vx))
            global_x_offset (float): Angle in degrees from East to global X-axis, 
                                   measured counterclockwise
                        
        Returns:
            float: Heading angle in degrees, clockwise from North
            
        For global_x_offset = 0°:
            [1, 0]   → 90° (East)
            [0, -1]  → 180° (South)
            [-1, 0]  → 270° (West)
            [0, 1]   → 0° (North)
            
        For global_x_offset = 90°:
            [1, 0]   → 0° (North)
            [0, -1]  → 90° (East)
            [-1, 0]  → 180° (South)
            [0, 1]   → 270° (West)
            
        For global_x_offset = 180°:
            [1, 0]   → 270° (West)
            [0, -1]  → 0° (North)
            [-1, 0]  → 90° (East)
            [0, 1]   → 180° (South)
            
        For global_x_offset = 270°:
            [1, 0]   → 180° (South)
            [0, -1]  → 270° (West)
            [-1, 0]  → 0° (North)
            [0, 1]   → 90° (East)
        """
        # Convert alpha from radians to degrees
        alpha_deg = degrees(alpha)
        alpha_deg = alpha_deg % 360
        
        # Calculate heading (theta):
        # 1. Subtract alpha_deg to convert from counterclockwise to clockwise
        # 2. Add 90 to start from North
        # 3. Subtract global_x_offset to account for frame rotation
        # 4. Use modulo to keep in [0, 360) range
        theta = (-alpha_deg + 90 - global_x_offset) % 360
        
        return theta
    
    def compute_heading_and_speed(self, velocity_x_and_y, global_x_offset, flag):
        """
        Compute heading and speed from velocity components.
        
        Args:
            velocity_x_and_y (numpy.array): [vx, vy] velocity components in rotated frame
            global_x_offset (float): Global X-axis offset from East (degrees counterclockwise)
            flag (str): Control flag ("green" for zero speed)
            
        Returns:
            tuple: (speed, heading) as strings with rounded values
        """
        # Calculate heading
        global_x_offset = global_x_offset % 360
        angle_rad = atan2(velocity_x_and_y[1], velocity_x_and_y[0])
        heading = self.alpha_to_theta(angle_rad, global_x_offset)
        
        # Calculate speed
        speed = np.sqrt(velocity_x_and_y[0]**2 + velocity_x_and_y[1]**2)
        speed = speed  # edit speed = self.map_speed(speed)
        
        # Apply flag condition
        if flag == "green":
            speed = 0
        

        return (f"{speed:.2f}", f"{heading:.2f}")
    
    def create_command(self, bot_id, timeout, speed, heading):
        return {
            "bot_id": bot_id,
            "pid_control": {
                "timeout": timeout,
                "speed": {"target": speed},
                "heading": {"target": heading}
            }
        }
    
    def send_command(self, command):
        response = requests.post(
            url=self.API_ENDPOINT_MANUAL,
            json=command,
            headers=self.headers
        )
        return response
    
    def control_bots(self, u_global_reshaped, global_x_offset, bot_ids, flag):
        print("\nSending Bot Commands:")
        print("-" * 50)
        
        for index, bot_id in enumerate(bot_ids):
            velocity = u_global_reshaped[:, index]
            speed, heading = self.compute_heading_and_speed(
                velocity, global_x_offset, flag
            )
                       
            command = self.create_command(
                bot_id=bot_id,
                timeout=4,
                speed=speed,
                heading=heading
            )
            
            # print(f"\nBot {bot_id}:")
            # print(f"Velocity: {velocity}")
            # print(f"Speed: {speed} m/s")
            # print(f"Heading: {heading}°")
            
            response = self.send_command(command)
            print(f"\nBot {bot_id} -> Command Response: {response.text}")
        
            time.sleep(0.2)
    
    def get_bot_data(self, u_global_reshaped, global_x_offset, bot_ids, flag):
        """
        Get bot data including ID, velocity, speed, and heading for each bot.
        
        Args:
            u_global_reshaped (numpy.array): 2xN array of velocities
            global_x_offset (float): Global X-axis offset in degrees
            bot_ids (list): List of bot IDs
            flag (str): Control flag
            
        Returns:
            list: List of dictionaries containing data for each bot
        """
        bot_data = []
        
        for index, bot_id in enumerate(bot_ids):
            velocity = u_global_reshaped[:, index]
            speed, heading = self.compute_heading_and_speed(
                velocity, global_x_offset, flag
            )
            
            bot_info = {
                'bot_id': bot_id,
                'velocity': velocity.tolist(),  # Convert numpy array to list
                'speed': float(speed),          # Convert string to float
                'heading': float(heading)       # Convert string to float
            }
            
            bot_data.append(bot_info)
            
        return bot_data


# Example usage
if __name__ == "__main__":
    # Input parameters
    hub_ip = "localhost:40001" # Hub IP address localhost:40001
    bot_ids = [1, 2, 3, 4]  # Bot IDs [1,2,3,4]
    global_x_offset = 0 # Global X-axis offset
    flag = "normal" # Control flag
    
    # Robot velocities
    robot_velocities = np.array([
        [1, -1, -1, 1], # x velocities
        [1, 1, -1, -1]  # y velocities
    ])
       
    # Initialize controller
    controller = JaiabotController(hub_ip=hub_ip)

    # Send waypoint commands to all bots
    controller.control_bots(robot_velocities, global_x_offset, bot_ids, flag)
    
    # Get bot data
    bot_data = controller.get_bot_data(robot_velocities, global_x_offset, bot_ids, flag)
    
    # Print the results
    print("\nBot Data:")
    print("-" * 50)
    for bot in bot_data:
        print(f"\nBot {bot['bot_id']}:")
        print(f"Velocity: [{bot['velocity'][0]}, {bot['velocity'][1]}]")
        print(f"Speed: {bot['speed']:.2f} m/s")
        print(f"Heading: {bot['heading']:.2f}°")


# import math
# import requests
# import json
# import time 
# import numpy as np
# from math import atan2, degrees
# from pyproj import CRS, Transformer
# import matplotlib.pyplot as plt

# # (A) Take control:
# # defining the api-endpoint
# API_ENDPOINT_ALL_STOP =  "http://localhost:40001/jaia/all-stop" # "http://10.23.10.12/jaia/all-stop" 
# API_ENDPOINT_MANUAL = "http://localhost:40001/jaia/ep-command" # "http://10.23.10.12/jaia/ep-command" 
# # API_ENDPOINT_ALL_STOP =  "http://10.23.10.11/jaia/all-stop" 
# # API_ENDPOINT_MANUAL = "http://10.23.10.11/jaia/ep-command" 

# # define the headers for the request
# headers = {'clientid': 'backseat-control', 'Content-Type' : 'application/json; charset=utf-8'}  

# # send all stop to take control
# all_stop_resp = requests.post(url=API_ENDPOINT_ALL_STOP, headers=headers)

# pastebin_url = all_stop_resp.text
# print("The pastebin URL is:%s"%pastebin_url) 


# class JaiabotController:
#     def __init__(self, hub_ip="10.23.10.11"):
#         self.API_ENDPOINT_ALL_STOP = f"http://{hub_ip}/jaia/all-stop"
#         self.API_ENDPOINT_MANUAL = f"http://{hub_ip}/jaia/ep-command"
        
#         self.headers = {
#             'clientid': 'backseat-control',
#             'Content-Type': 'application/json; charset=utf-8'
#         }
        
#         self.speed_params = {
#             'max_speed': 3.0,
#             'min_speed': 0.0,
#             'max_speed_org': 3.0,
#             'max_safety_speed': 2.0,
#             'min_safety_speed': 0.0
#         }
    
#     def take_control(self):
#         response = requests.post(
#             url=self.API_ENDPOINT_ALL_STOP,
#             headers=self.headers
#         )
#         print(f"All Stop Response: {response.text}")
#         return response.ok
    
#     @staticmethod
#     def clamp(value, min_val, max_val):
#         return max(min_val, min(max_val, value))
    
#     def map_speed(self, speed):
#         clamped_speed = self.clamp(speed, 0, self.speed_params['max_speed_org'])
#         normalized_speed = clamped_speed / self.speed_params['max_speed_org']
#         mapped_speed = (self.speed_params['min_speed'] + 
#                        normalized_speed * (self.speed_params['max_speed'] - 
#                                         self.speed_params['min_speed']))
#         limited_speed = self.clamp(mapped_speed,
#                                  self.speed_params['min_safety_speed'],
#                                  self.speed_params['max_safety_speed'])
#         return limited_speed
    
#     @staticmethod
#     def alpha_to_theta(alpha, global_x_offset):
#         """
#         Convert angle from global reference frame (alpha) to compass heading (theta).
#         Global reference frame is right-hand oriented.
        
#         Args:
#             alpha (float): Angle in radians from global frame (atan2(vy, vx))
#             global_x_offset (float): Angle in degrees from East to global X-axis, 
#                                    measured counterclockwise
                        
#         Returns:
#             float: Heading angle in degrees, clockwise from North
            
#         For global_x_offset = 0°:
#             [1, 0]   → 90° (East)
#             [0, -1]  → 180° (South)
#             [-1, 0]  → 270° (West)
#             [0, 1]   → 0° (North)
            
#         For global_x_offset = 90°:
#             [1, 0]   → 0° (North)
#             [0, -1]  → 90° (East)
#             [-1, 0]  → 180° (South)
#             [0, 1]   → 270° (West)
            
#         For global_x_offset = 180°:
#             [1, 0]   → 270° (West)
#             [0, -1]  → 0° (North)
#             [-1, 0]  → 90° (East)
#             [0, 1]   → 180° (South)
            
#         For global_x_offset = 270°:
#             [1, 0]   → 180° (South)
#             [0, -1]  → 270° (West)
#             [-1, 0]  → 0° (North)
#             [0, 1]   → 90° (East)
#         """
#         # Convert alpha from radians to degrees
#         alpha_deg = degrees(alpha)
        
#         # Calculate heading:
#         # 1. Subtract alpha_deg to convert from counterclockwise to clockwise
#         # 2. Add 90 to start from North
#         # 3. Subtract global_x_offset to account for frame rotation
#         # 4. Use modulo to keep in [0, 360) range
#         theta = (-alpha_deg + 90 - global_x_offset) % 360
        
#         return theta
    
#     def compute_heading_and_speed(self, velocity_x_and_y, global_x_offset, flag):
#         """
#         Compute heading and speed from velocity components.
        
#         Args:
#             velocity_x_and_y (numpy.array): [vx, vy] velocity components in rotated frame
#             global_x_offset (float): Global X-axis offset from East (degrees counterclockwise)
#             flag (str): Control flag ("green" for zero speed)
            
#         Returns:
#             tuple: (speed, heading) as strings with rounded values
#         """
#         # Calculate heading
#         global_x_offset = global_x_offset % 360
#         angle_rad = atan2(velocity_x_and_y[1], velocity_x_and_y[0])
#         heading = self.alpha_to_theta(angle_rad, global_x_offset)
        
#         # Calculate speed
#         speed = np.sqrt(velocity_x_and_y[0]**2 + velocity_x_and_y[1]**2)
#         speed = self.map_speed(speed)
        
#         # Apply flag condition
#         if flag == "green":
#             speed = 0
            
#         return (f"{speed:.2f}", f"{heading:.2f}")
    
#     def create_command(self, bot_id, timeout, speed, heading):
#         return {
#             "bot_id": bot_id,
#             "pid_control": {
#                 "timeout": timeout,
#                 "speed": {"target": speed},
#                 "heading": {"target": heading}
#             }
#         }
    
#     def send_command(self, command):
#         response = requests.post(
#             url=self.API_ENDPOINT_MANUAL,
#             json=command,
#             headers=self.headers
#         )
#         return response
    
#     def control_bots(self, u_global_reshaped, global_x_offset, bot_ids, flag):
#         print("\nSending Bot Commands:")
#         print("-" * 50)
        
#         for index, bot_id in enumerate(bot_ids):
#             velocity = u_global_reshaped[:, index]
#             speed, heading = self.compute_heading_and_speed(
#                 velocity, global_x_offset, flag
#             )
            
#             command = self.create_command(
#                 bot_id=bot_id,
#                 timeout=3,
#                 speed=speed,
#                 heading=heading
#             )
            
#             print(f"\nBot {bot_id}:")
#             print(f"Velocity: {velocity}")
#             print(f"Speed: {speed} m/s")
#             print(f"Heading: {heading}°")
            
#             response = self.send_command(command)
#             print(f"Command Response: {response.text}")
        
#         time.sleep(0.1)

# # Test the robot control with multiple bots
# if __name__ == "__main__":
#     # Input parameters
#     hub_ip = "localhost:40001"  # Hub IP address
#     bot_ids = [1, 2, 3, 4]  # Bot IDs
#     global_x_offset = 0     # Global X-axis offset
#     flag = "normal"         # Control flag
    
#     # Robot velocities as 2x4 numpy array
#     # First row: x velocities
#     # Second row: y velocities
#     robot_velocities = np.array([
#         [1, -1, -1, 1],      # x velocities
#         [1, 1, -1, -1]       # y velocities
#     ])
    
#     # Initialize controller with hub IP
#     controller = JaiabotController(hub_ip=hub_ip)
    
#     # Take control first
#     controller.take_control()
    
#     # Control the bots
#     controller.control_bots(robot_velocities, global_x_offset, bot_ids, flag)