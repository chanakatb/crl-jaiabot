import numpy as np
import matplotlib.pyplot as plt
import time

def plot_robot_trajectories_animated(filename):
   data = np.load(filename)
   num_points = data.shape[2]
   
   plt.ion()  # Turn on interactive mode
   fig = plt.figure(figsize=(10,8))
   colors = ['r', 'g', 'b', 'm']
   
   for t in range(num_points):
       plt.clf()  # Clear figure
       
       # Plot complete trajectories up to current time
       for i in range(4):
           plt.plot(data[1,i,:t+1], data[0,i,:t+1], 
                   color=colors[i], 
                   label=f'Bot {i+1}',
                   marker='o',
                   markersize=4,
                   linestyle='-',
                   linewidth=1)
           
           # Start points
           plt.plot(data[1,i,0], data[0,i,0], 
                   color=colors[i],
                   marker='*', 
                   markersize=15,
                   markerfacecolor='white')
           
           # End points if at final time step
           if t == num_points-1:
               plt.plot(data[1,i,-1], data[0,i,-1], 
                       color=colors[i],
                       marker='s', 
                       markersize=10,
                       markerfacecolor='white')

       # Legend markers
       plt.plot([], [], 
               color='k',
               marker='*', 
               markersize=15,
               markerfacecolor='white',
               linestyle='None',
               label='Start')
       
       plt.plot([], [], 
               color='k',
               marker='s', 
               markersize=10,
               markerfacecolor='white',
               linestyle='None',
               label='End')

       plt.xlabel('Longitude (°)')
       plt.ylabel('Latitude (°)') 
       plt.title(f"Robots predicted Lat & Lon- Step {t+1}/{num_points}")
       plt.legend()
       plt.grid(True)
       
       print(f"\nStep {t+1}/{num_points}")
       for i in range(4):
           print(f"Bot {i+1}: Lat={data[0,i,t]:.6f}°, Lon={data[1,i,t]:.6f}°")
       
       plt.pause(0.5)
       
   plt.ioff()  # Turn off interactive mode
   plt.show()

if __name__ == "__main__":
   plot_robot_trajectories_animated('data/20250122_153351_robots_lat_lon_predicted.npy')

   