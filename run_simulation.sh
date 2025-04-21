#!/bin/bash

# Navigate to the initial directory (if needed, specify the path correctly)
cd /home/ubuntu20 || { echo "Failed to change directory to /home/ubuntu20"; exit 1; }

# Navigate to the desired subdirectory
cd jaiabot/src/web/ || { echo "Failed to change directory to jaiabot/src/web/"; exit 1; }

# Run the simulator script in the background
./run.sh &

# Wait for 80 seconds
echo "NOTE: Wait until the modules load ..."
sleep 80

# Launch the simulator in a new terminal
gnome-terminal -- bash -c "
    cd /home/ubuntu20/jaiabot/config/launch/simulation || { echo 'Failed to change directory to /home/ubuntu20/jaiabot/config/launch/simulation'; exit 1; }
    ./generate_all_launch.sh 4 1 || { echo 'Failed to execute generate_all_launch.sh'; exit 1; }
    ./all.launch || { echo 'Failed to execute all.launch'; exit 1; }
    exec bash
" || { echo "Failed to open new terminal and execute commands"; exit 1; }

echo "All commands executed successfully."






