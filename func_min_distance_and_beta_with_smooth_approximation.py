import numpy as np
import sys
from sympy import symbols, diff, exp, log, sqrt, sympify, Symbol


def calculate_exp_sum(distances, rho):
    """
    Calculate the sum of exponentials: sum(exp(-rho * d_i)) for all distances
    using vectorized operations for better performance.
    
    Parameters:
    ----------
    distances : list or numpy.ndarray
        List of distance values
    rho : float
        Scaling parameter
        
    Returns:
    -------
    float
        The sum of all exponential terms
    """
    # Convert to numpy array if not already
    d_array = np.array(distances)    
    # Calculate all exponential terms at once
    exp_terms = np.exp(-rho*d_array)
    # print("exp_terms: \n", exp_terms)
    
    # Sum all terms
    return np.sum(exp_terms)

def min_distance_and_beta_with_smooth_approx(B_matrix, rho, mu_nf, a_nf, r_nf, *args):
    # Ensure that we have enough coordinates for the matrix
    num_points = B_matrix.shape[1]  # Number of columns in B_matrix
    
    if len(args) < 2 * num_points:
        raise ValueError(f"Not enough coordinates provided. B_matrix has {num_points} points, but only {len(args)//2} points' coordinates were provided.")
    
    # Convert args into x and y coordinates
    x = args[::2]  # even indices are x coordinates
    y = args[1::2]  # odd indices are y coordinates
    
    # # Create symbolic variables
    # x_sym = [symbols(f'x{i}') for i in range(num_points)]  # Use num_points instead of len(x)
    # y_sym = [symbols(f'y{i}') for i in range(num_points)]  # Use num_points instead of len(y)
    # d = [symbols(f'd{i}') for i in range(B_matrix.shape[0])]
    # rho_sym = symbols('rho')
    
    distances = []
    all_distances = {}
    symbolic_distances = {}
    
    # Calculate all distances
    for i in range(B_matrix.shape[0]):
        indices = np.nonzero(B_matrix[i])[0]
        
        idx1, idx2 = None, None
        for j in indices:
            if B_matrix[i][j] == 1:
                idx1 = j
            elif B_matrix[i][j] == -1:
                idx2 = j
        
        if idx1 is not None and idx2 is not None:
            # Check if indices are within bounds
            if idx1 >= num_points or idx2 >= num_points:
                print(f"Warning: Index out of range for row {i}. idx1={idx1}, idx2={idx2}, num_points={num_points}")
                continue
                
            # # Create symbolic distance expression
            # d_symbolic = sqrt(((x_sym[idx1] - x_sym[idx2]))**2 + 
            #                 ((y_sym[idx1] - y_sym[idx2]))**2)
            
            # Calculate actual distance
            d_val = np.sqrt(((x[idx1] - x[idx2]))**2 + 
                           ((y[idx1] - y[idx2]))**2)
            
            distances.append(d_val)
            all_distances[f'd{i}'] = d_val
            # symbolic_distances[i] = d_symbolic

   
    # Create substitution dictionaries for numerical evaluation
    coord_values = {}
    for i in range(num_points):
        if i < len(x):  # Make sure we don't go out of bounds
            coord_values[Symbol(f'x{i}')] = x[i]
            coord_values[Symbol(f'y{i}')] = y[i]
    
    distance_values = {}
    for i, d_val in all_distances.items():
        distance_values[Symbol(i)] = d_val
  
    # Calculate minimum distance and beta value
    if not distances:
        raise ValueError("No valid distances calculated")
    
    # print("distances: \n", np.array(distances))
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    # print("\n\nmin_dist: ", min_dist)

    # ------- correction -------
    exp_sum = calculate_exp_sum(distances, rho)
    min_dist_approximation = (-1/rho) * np.log(exp_sum)
    # print("min_dist_approximation: ", min_dist_approximation)
    
    # Define beta function - using numpy functions consistently
    beta_value = np.log(mu_nf - a_nf * np.exp(-(-r_nf + min_dist_approximation + min_dist_approximation**2)**2))
    # print("beta_value: ", beta_value)

    # ∂β/∂d
    dbeta_dd = -(2*a_nf*(1+2*min_dist_approximation)*(min_dist_approximation+min_dist_approximation**2-r_nf))/(a_nf-mu_nf*np.exp((min_dist_approximation+min_dist_approximation**2-r_nf)**2))
    # print("∂β/∂d: ", dbeta_dd)
    # print("Σ exp(-ρ di): ", exp_sum)
    
    # Calculate gradient beta
    gradient_beta = np.zeros(2 * num_points)
    
    for i in range(num_points):
        dx_sum = 0
        dy_sum = 0
        
        for j in range(B_matrix.shape[0]):
            indices = np.nonzero(B_matrix[j])[0]
            
            idx1, idx2 = None, None
            for k in indices:
                if B_matrix[j][k] == 1:
                    idx1 = k
                elif B_matrix[j][k] == -1:
                    idx2 = k
            
            if idx1 is not None and idx2 is not None and f'd{j}' in all_distances:
                d_val = all_distances[f'd{j}']
                exp_term = np.exp(-rho * d_val)
                
                if idx1 == i and idx1 < len(x) and idx2 < len(x):
                    dx_sum += exp_term * ((x[idx1] - x[idx2])) / d_val
                    dy_sum += exp_term * ((y[idx1] - y[idx2])) / d_val
                elif idx2 == i and idx1 < len(x) and idx2 < len(x):
                    dx_sum += exp_term * ((x[idx2] - x[idx1])) / d_val
                    dy_sum += exp_term * ((y[idx2] - y[idx1])) / d_val
        
        if i < len(x):  # Make sure we don't go out of bounds
            gradient_beta[2*i] = dbeta_dd * ((1/exp_sum) * dx_sum)
            gradient_beta[2*i + 1] = dbeta_dd * ((1/exp_sum) * dy_sum)
    
    return min_dist_approximation, min_idx, beta_value, gradient_beta,  all_distances

def main():
    B = np.array([
        [ 1, -1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1, -1,  0,  0,  0,  0,  0,  0],
        [ 0,  0, -1,  1,  0,  0,  0,  0,  0],
        [ 0,  1,  0, -1,  0,  0,  0,  0,  0],
        [-1,  0,  1,  0,  0,  0,  0,  0,  0],
        [ 1,  0,  0, -1,  0,  0,  0,  0,  0],
        [ 1,  0,  0,  0, -1,  0,  0,  0,  0],
        [ 0,  1,  0,  0, -1,  0,  0,  0,  0],
        [ 0,  0,  1,  0, -1,  0,  0,  0,  0],
        [ 0,  0,  0,  1, -1,  0,  0,  0,  0],
        [ 1,  0,  0,  0,  0, -1,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  0, -1,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0, -1,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0, -1]
    ])
    
    # Make sure to provide all 9 points (x0,y0 through x8,y8)
    coords = [0, 0,     # x0, y0
              1, 1,     # x1, y1
              2, 2,     # x2, y2
              7.8, 7.5, # x3, y3
              4, 4,     # x4, y4
              5, 5,     # x5, y5
              6, 6,     # x6, y6
              7, 7,     # x7, y7
              8, 8]     # x8, y8
    scale = 0.005
    coords = [coord *  scale for coord in coords]
    # Check if we have enough coordinates
    if len(coords) < 2 * B.shape[1]:
        print(f"Warning: Not enough coordinates. Needed {2 * B.shape[1]}, but got {len(coords)}")
    
    rho = 1000 #10 or 2000
    a_nf = 2.7182818  
    mu_nf = 2    
    r_nf = -1   
    
    min_dist, min_idx, beta_value, gradient_beta, all_distances = min_distance_and_beta_with_smooth_approx(
        B, rho, mu_nf, a_nf, r_nf, *coords)
    
    print("\nDistance values:")
    for i in range(B.shape[0]):  # Use B.shape[0] instead of hardcoded 14
        if f'd{i}' in all_distances:
            print(f"d[{i}] = {all_distances[f'd{i}']}")
    
    print("Minimum distance:", min_dist)
    print("Index of minimum distance (d{})".format(min_idx))
    print("\nBeta value:", beta_value)
    
    print("\nNumerical Gradient beta:")
    for i in range(len(gradient_beta)//2):
        print("dβ/dx{}: {}".format(i, gradient_beta[2*i]))
        print("dβ/dy{}: {}".format(i, gradient_beta[2*i + 1]))
    

if __name__ == "__main__":
    main()


