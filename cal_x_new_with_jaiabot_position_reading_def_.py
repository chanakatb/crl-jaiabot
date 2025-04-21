import numpy as np

def cal_x_new_with_jaiabot_position_reading_(x0_, robots_position_new, no_of_robots, no_of_obstacles):
    # Define zero matrix
    x_new = np.zeros((2, x0_.shape[1]))
    for i in range(0, x0_.shape[1]): 
        if i < no_of_robots:
            # --- Robots' position and Boundaries' position ---
            x_new[:, [i]] = robots_position_new[:, [i]]
        elif i >= no_of_robots and i < no_of_robots + no_of_obstacles:
            # --- Obstacles' position --- 
            # Modify this as we wish
            x_new[:, [i]] =  x0_[:, [i]]
        else:
            # ---  ---
            # Modify this as we wish
            pass
                         

    return x_new
