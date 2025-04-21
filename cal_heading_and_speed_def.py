import numpy as np

def cal_heading_and_speed(velocity_array, bot_ids):
    """
    Calculate heading and speed for each bot and return as a list of dictionaries.
    
    Args:
        velocity_array (np.ndarray): Array of shape (2, n) where each column contains [x_vel, y_vel]
                                   x_vel is positive towards East
                                   y_vel is positive towards North
        bot_ids (list): List of bot identifiers
    
    Returns:
        list: List of dictionaries containing bot data with keys:
              'bot_id': bot identifier
              'velocity': [x_vel, y_vel] array
              'speed': magnitude of velocity
              'heading': heading in degrees clockwise from North
    """
    # Number of velocity vectors
    n_vectors = velocity_array.shape[1]
    
    # Initialize list to store bot data
    bot_data = []
    
    for i in range(n_vectors):
        x_vel = velocity_array[0, i]  # East component
        y_vel = velocity_array[1, i]  # North component
        
        # Calculate speed (magnitude of velocity vector)
        speed = np.sqrt(x_vel**2 + y_vel**2)
        
        # Calculate heading (90° - arctan2 gives angle from North clockwise)
        heading = 90 - np.degrees(np.arctan2(y_vel, x_vel))
        
        # Ensure heading is between 0 and 360 degrees
        if heading < 0:
            heading += 360
            
        # Create dictionary for current bot
        bot_dict = {
            'bot_id': bot_ids[i],
            'velocity': np.array([x_vel, y_vel]),
            'speed': speed,
            'heading': heading
        }
        
        bot_data.append(bot_dict)
    
    return bot_data
# Example usage
if __name__ == "__main__":

    command_velocity = np.array([
        [-1.37079914, 1.39648301, -0.60214652, 1.17705252],
        [0.27984846, -0.46522398, -1.31961841, 0.86901332]
    ])

    command_velocity = np.array([
        [1,  1, -1, -1],
        [1, -1, -1,  1]
    ])

    # Example bot IDs
    bot_ids = [1, 2, 3, 4]  # You can modify these as needed

    # Calculate heading and speed for each bot
    bot_data = cal_heading_and_speed(command_velocity, bot_ids)

    # Print results in the requested format
    for bot in bot_data:
        print(f"\nBot {bot['bot_id']}:")
        print(f"command_velocity: [{bot['velocity'][0]:.2f}, {bot['velocity'][1]:.2f}]")
        print(f"Speed: {bot['speed']:.2f} m/s")
        print(f"Heading: {bot['heading']:.2f}°")
