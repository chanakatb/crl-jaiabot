import numpy as np
from cal_nearest_point_on_ellipse_boundary_all_cal_def import cal_nearest_point_to_boundary

def cal_nearest_points_to_bots(x, indexForRelativePositions, noOfRelativeDistancesForBots, noOfRelativeDistancesForObstacles,origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse):
    # initialization   
    grad_phi_min_index = np.zeros((2 * len(indexForRelativePositions), 1))
    # index_for_relative_positions = np.array([[0, 1], [1, 2], [3, 2], [1, 3], [2, 0], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8]])
    nearest_points = []
    for i in range(len(indexForRelativePositions)):
        # print(f"i: {i}")
        if i < noOfRelativeDistancesForBots:
            pass

        elif i < noOfRelativeDistancesForBots + noOfRelativeDistancesForObstacles: # check this number always
            pass
            
        else:
            Xi = x[0, indexForRelativePositions[i, 0]]
            Yi = x[1, indexForRelativePositions[i, 0]]       

            nearest_point, min_distance = cal_nearest_point_to_boundary(Xi, Yi, origin_x_ellipse, origin_y_ellipse, major_axis_ellipse, minor_axis_ellipse, rotational_angle_ellipse)
            nearest_points.append(nearest_point)
    nearest_points = np.array(nearest_points)
    return (nearest_points.T).squeeze()
