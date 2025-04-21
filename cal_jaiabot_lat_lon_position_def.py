import json
import time
import requests
import numpy as np
from cal_gps_to_local_coordinates_def import gps_to_local_coordinates

def cal_jaiabot_lat_lon_position(bot_ids, api_endpoint=None):
    """
    Calculate the latitude and longitude positions for specified Jaiabots
    
    Args:
        bot_ids (list): List of bot IDs as integers (e.g., [2, 6, 7, 8])
        api_endpoint (str, optional): API endpoint URL. If None, uses simulator endpoint.
                                    Examples:
                                    - "http://localhost:40001/jaia/status" (simulator)
                                    - "http://10.23.10.11/jaia/status" (Hub 1)
                                    - "http://10.23.10.12/jaia/status" (Hub 2)
    
    Returns:
        tuple: (robots_lat_lon, hub_lat_lon)
            robots_lat_lon (numpy.ndarray): 2xN matrix containing lat/lon positions for each bot
            hub_lat_lon (numpy.ndarray): 2x1 matrix containing hub lat/lon position
    """
    # Define zero matrix for the number of bots provided
    no_of_robots = len(bot_ids)
    robots_lat_lon = np.zeros((2, no_of_robots))
    robots_heading = np.zeros((1, no_of_robots))
    robots_speed = np.zeros((1, no_of_robots))
    hub_lat_lon = np.zeros((2, 1))
    
    # Set default API endpoint if none provided
    if api_endpoint is None:
        api_endpoint = "http://localhost:40001/jaia/status"  # simulator default
        # api_endpoint = "http://10.23.10.11/jaia/status" # change
    
    # Define the headers for the request
    headers = {'clientid': 'hub-button-all-stop'}
    
    try:
        # Send request and get response
        status = requests.get(url=api_endpoint, headers=headers, timeout=30)
        status.raise_for_status()  # Raise exception for bad status codes
        
        # Parse response
        data = json.loads(status.text)
        
        # Get bot positions
        for i, bot_id in enumerate(bot_ids):
            try:
                bot_id_str = str(bot_id)
                robots_lat_lon[:, [i]] = np.array([
                    [data["bots"][bot_id_str]["location"]["lat"]],
                    [data["bots"][bot_id_str]["location"]["lon"]]
                ])
                robots_heading[:, [i]] = np.array([
                    [data["bots"][bot_id_str]["attitude"]["heading"]]
                ])

                robots_speed[:, [i]] = np.array([
                    [data["bots"][bot_id_str]["speed"]["over_ground"]]
                ])
            except KeyError:
                print(f"Warning: Bot ID {bot_id} not found in status data")
        
        # Get hub position
        hub_lat_lon[:,[0]] = np.array([
            [39.90166],
            [-76.16750]
        ]) # change
        # hub_lat_lon[:,[0]] = np.array([
        #     [data["hubs"]["1"]["location"]["lat"]],
        #     [data["hubs"]["1"]["location"]["lon"]]
        # ]) # change
        
    except requests.RequestException as e:
        print(f"Error accessing API at {api_endpoint}: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise
        
    return robots_lat_lon, hub_lat_lon, robots_heading, robots_speed

# Example usage
if __name__ == "__main__":
    # Define the bot IDs
    bot_ids = [1, 2, 3, 4]
    
    global_x_offset = 0

    # Example endpoints
    endpoints = {
        "simulator": "http://localhost:40001/jaia/status",
        # "hub1": "http://10.23.10.11/jaia/status",
        # "hub2": "http://10.23.10.12/jaia/status"
    }
    
    # Test with different endpoints
  
    for name, endpoint in endpoints.items():
        print(f"\nTesting with {name} endpoint: {endpoint}")
        print("-" * 50)
                   
        try:
            robots_lat_lon, hub_lat_lon, robots_heading = cal_jaiabot_lat_lon_position(
                bot_ids=bot_ids,
                api_endpoint=endpoint
            )
            
            

            for i, bot_id in enumerate(bot_ids):
                print(f"Bot {bot_id}:")
                print(f"  Latitude:  {robots_lat_lon[0,i]:.6f}")
                print(f"  Longitude: {robots_lat_lon[1,i]:.6f}")
            
            print("\nHub Position:")
            print(f"  Latitude:  {hub_lat_lon[0,0]:.6f}")
            print(f"  Longitude: {hub_lat_lon[1,0]:.6f}")
            
        except Exception as e:
            print(f"Failed to get data from {name}: {str(e)}")
            continue
    
    print('robots_lat_lon: \n', robots_lat_lon)
    print('hub_lat_lon: \n', hub_lat_lon)
    print('robots_heading: \n', robots_heading)
    latitude_hub, longitude_hub = hub_lat_lon.flatten()
    robots_position = gps_to_local_coordinates(robots_lat_lon, latitude_hub, longitude_hub, global_x_offset=0)
    print('Robots position: \n', robots_position)