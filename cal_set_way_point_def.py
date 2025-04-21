import requests
import numpy as np
import time

class JaiabotWaypointController:
    def __init__(self, hub_ip="localhost:40001", api_endpoint_all_stop=None, api_endpoint_waypoint=None):
        """
        Initialize the Jaiabot waypoint controller.
        
        Args:
            hub_ip (str): IP address of the hub
            api_endpoint_all_stop (str, optional): Full URL for the all-stop endpoint
            api_endpoint_waypoint (str, optional): Full URL for the waypoint mission endpoint
        """
        # Use provided endpoints if available, otherwise construct them from hub_ip
        self.API_ENDPOINT_ALL_STOP = api_endpoint_all_stop if api_endpoint_all_stop else f"http://{hub_ip}/jaia/all-stop"
        self.API_ENDPOINT_WAYPOINT = api_endpoint_waypoint if api_endpoint_waypoint else f"http://{hub_ip}/jaia/single-waypoint-mission"
        
        self.headers = {
            'clientid': 'backseat-control',
            'Content-Type': 'application/json; charset=utf-8'
        }
    
    def take_control_(self):
        """Take control of the bots by sending all-stop command."""
        response = requests.post(
            url=self.API_ENDPOINT_ALL_STOP,
            headers=self.headers
        )
        print(f"All Stop Response: {response.text}")
        return response.ok
    
    def create_waypoint_command_(self, bot_id, lat, lon):
        """
        Create a waypoint command for a single bot.
        
        Args:
            bot_id (int): ID of the bot
            lat (float): Target latitude
            lon (float): Target longitude
            
        Returns:
            dict: Waypoint command data
        """
        return {
            "bot_id": bot_id,
            "lat": float(lat),
            "lon": float(lon)
        }
    
    def send_waypoint_command_(self, command):
        """
        Send a waypoint command to a bot.
        
        Args:
            command (dict): Waypoint command data
            
        Returns:
            requests.Response: Response from the API
        """
        response = requests.post(
            url=self.API_ENDPOINT_WAYPOINT,
            json=command,
            headers=self.headers
        )
        return response
    
    def control_bots_(self, robots_lat_lon, bot_ids, flag=None):
        """
        Send waypoint commands to multiple bots.
        
        Args:
            robots_lat_lon (numpy.array): 2x4 array with [latitude, longitude] for each bot
            bot_ids (list): List of bot IDs
            flag (str): Control flag ("green" to skip sending commands)
        """
        if flag == "green":
            print("Flag is green - skipping waypoint commands")
            return
        
        print("\n\nSending Bot Waypoint Commands:")
        print("-" * 50)
        
        for index, bot_id in enumerate(bot_ids):
            operation_start_time = time.time()
            lat, lon = robots_lat_lon[:, index]            
            command = self.create_waypoint_command_(
                bot_id=bot_id,
                lat=lat,
                lon=lon
            )
            
            print(f"\nBot {bot_id}:")
            print(f"Target Position: Lat={lat:.6f}, Lon={lon:.6f}")
            
            response = self.send_waypoint_command_(command)
            print(f"Command Response: {response.text}")

            # Calculate and print time taken for the operation
            operation_end_time = time.time()
            operation_time = operation_end_time - operation_start_time
            print(f"Time duration to send waypoint command: {operation_time:.4f} seconds")
            
            # Add small delay between commands
            time.sleep(0.2)
    
    def get_bot_data_(self, robots_lat_lon, bot_ids, flag=None):
        """
        Get bot data including ID and target positions.
        
        Args:
            robots_lat_lon (numpy.array): 2x4 array with [latitude, longitude] for each bot
            bot_ids (list): List of bot IDs
            flag (str): Control flag
            
        Returns:
            list: List of dictionaries containing data for each bot
        """
        bot_data = []
        
        for index, bot_id in enumerate(bot_ids):
            lat, lon = robots_lat_lon[:, index]
            
            bot_info = {
                'bot_id': bot_id,
                'target_position': {
                    'latitude': float(lat),
                    'longitude': float(lon)
                }
            }
            
            bot_data.append(bot_info)
            
        return bot_data


# Example usage
if __name__ == "__main__":
    for i in range(50):       
                                                                                                
        # Input parameters
        hub_ip = "localhost:40001"  # Hub IP address: localhost:40001 or 10.23.10.11
        
        # Define full API endpoints
        all_stop_endpoint = f"http://{hub_ip}/jaia/all-stop"
        waypoint_endpoint = f"http://{hub_ip}/jaia/single-waypoint-mission"  # Fixed: removed duplicate "jaia/"
        
        bot_ids = [1, 2, 3, 4]  # Bot IDs
        
        # Example waypoints for 4 bots (2x4 array: [latitudes; longitudes])
        robots_lat_lon = np.array([
            [39.901255, 39.90166, 39.90166, 39.90166],  # Latitudes
            [-76.168597, -76.16750, -76.16750, -76.16750]  # Longitudes
        ])
        
        # Initialize controller with full API endpoints
        controller = JaiabotWaypointController(
            hub_ip=hub_ip,
            api_endpoint_all_stop=all_stop_endpoint,
            api_endpoint_waypoint=waypoint_endpoint
        )
        
        # Send waypoint commands to all bots
        controller.control_bots_(robots_lat_lon, bot_ids)
        
        # Get and print bot data
        bot_data = controller.get_bot_data_(robots_lat_lon, bot_ids)

        # sleep
        time.sleep(0.2)