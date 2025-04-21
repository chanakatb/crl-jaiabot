import time

class BotCommandTimer:
    def __init__(self, interval):
        """
        Initialize the timer with a specified interval
        
        Args:
            interval (float): Time in seconds between bot commands
        """
        self.interval = interval
        self.last_command_time = 0
        
    def can_send_command(self):
        """
        Check if enough time has passed to send another command
        
        Returns:
            bool: True if it's time to send a command, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_command_time >= self.interval:
            self.last_command_time = current_time
            return True
        return False