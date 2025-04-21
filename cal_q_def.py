import numpy as np
from cal_nearest_point_on_ellipse_boundary_all_cal_def import cal_nearest_point_to_boundary

def cal_q(x, indexForRelativePositions, no_of_relative_distances_for_bots, no_of_relative_distances_for_obstacles, scaling_factor, center_x, center_y, a, b, phi):
    num_rows = len(indexForRelativePositions)
    q = np.zeros((2 * num_rows, 1))

    for i in range(num_rows):
        if i < no_of_relative_distances_for_bots + no_of_relative_distances_for_obstacles: # check this number
            x_i = x[0, indexForRelativePositions[i, 0]]
            y_i = x[1, indexForRelativePositions[i, 0]]
            x_j = x[0, indexForRelativePositions[i, 1]]
            y_j = x[1, indexForRelativePositions[i, 1]]
            q[2 * i:2 * (i + 1)] = scaling_factor * np.array([[x_j - x_i], [y_j - y_i]])
        else:
            x_i = x[0, indexForRelativePositions[i, 0]]
            y_i = x[1, indexForRelativePositions[i, 0]]
            nearest_point, min_distance = cal_nearest_point_to_boundary(x_i, y_i, center_x, center_y, a, b, phi)
            x_j = nearest_point[0, 0]
            y_j = nearest_point[1, 0]
            # print(f"The nearest points from Robot {i-7} to the boundary: {x_j, y_j}")
            q[2 * i:2 * (i + 1)] = scaling_factor * np.array([[x_j - x_i], [y_j - y_i]])      
            # print(f"i: {i} -> (xi, yi) = ({x_i}, {y_i})     (xj, yj) = ({x_j}, {y_j})") 
    
    return np.round(q,4)