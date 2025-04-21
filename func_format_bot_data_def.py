def format_bot_data(bot_ids, robots_current_heading, robots_current_speed):
    """
    Formats bot data into a list of dictionaries containing bot_id, speed, and heading.
    
    Parameters:
        bot_ids (list): List of bot IDs.
        robots_current_heading (list of lists): Nested list containing headings for each bot.
        robots_current_speed (list of lists): Nested list containing speeds for each bot.
    
    Returns:
        list: Formatted bot data.
    """
    bot_data = []
    for i, bot_id in enumerate(bot_ids):
        bot_info = {
            "bot_id": bot_id,
            "heading": robots_current_heading[0][i],  # Extract heading value
            "speed": robots_current_speed[0][i]       # Extract speed value
        }
        bot_data.append(bot_info)
    
    return bot_data
# Example usage
if __name__ == "__main__":
    # Given bot information
    bot_ids = [2, 6, 7, 8]
    robots_current_heading = [[48., 6., 236., 6.]]  # 2D list
    robots_current_speed = [[2, 5, 0.3, 0.09]]      # 2D list

    # Format the data
    bot_data = format_bot_data(bot_ids, robots_current_heading, robots_current_speed)

    # Print the formatted bot data
    for bot in bot_data:
        print(f"\nBot {bot['bot_id']}:")
        # print(f"Velocity: [{bot['velocity'][0]:.2f}, {bot['velocity'][1]:.2f}]")  # Commented out since velocity is not provided
        print(f"Speed: {bot['speed']:.2f} m/s")
        print(f"Heading: {bot['heading']:.2f}°")
