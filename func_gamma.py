import numpy as np
from sympy import symbols, diff, expand

def gamma_value_and_gradient(B_gamma, B, desired_relative_positions, coords_predicted_scaled):
    """
    Calculate symbolic gamma function, evaluate it and calculate gradients at given coordinates
    
    Args:
        B_gamma: 2D numpy array representing the B matrix used to create gamma function
        B: 2D numpy array representing the B matrix used for gradient calculations
        desired_relative_positions: desired relative positions of the points
        coords_predicted_scaled: list of coordinate values [x0,y0,x1,y1,...] to evaluate at
        
    Returns:
        gamma_value: evaluated value of gamma function
        gamma_gradient: evaluated gradient of gamma function
        symbolic_derivatives: symbolic derivatives of gamma function
    """
    # Step 1: Calculate symbolic gamma function from B_gamma matrix
    num_columns = B_gamma.shape[1]
    
    # Dynamically create symbolic variables for x and y
    x = symbols(f'x0:{num_columns}')
    y = symbols(f'y0:{num_columns}')
    
    # Calculate qx and qy through matrix multiplication
    qx = []
    qy = []
    
    # For each row in B_gamma
    for i in range(B_gamma.shape[0]):
        qx_term = 0
        qy_term = 0
        # For each column
        for j in range(num_columns):
            qx_term += B_gamma[i, j] * x[j]
            qy_term += B_gamma[i, j] * y[j]
        qx.append(qx_term)
        qy.append(qy_term)
    
    # Calculate gamma as sum of squares
    gamma_symbolic_function = 0
    for i in range(len(qx)):
        gamma_symbolic_function += (qx[i] - desired_relative_positions[i, 0])**2 + (qy[i] - desired_relative_positions[i, 1])**2
    
    gamma_symbolic_function = expand(gamma_symbolic_function)  # Expand the expression
    
    # Step 2: Calculate gradients for the symbolic gamma function
    # Get number of columns from B matrix (number of points)
    num_points = B.shape[1]
    
    # Create symbols for all possible points
    vars_dict = {}
    for i in range(num_points):
        vars_dict[f'x{i}'] = symbols(f'x{i}')
        vars_dict[f'y{i}'] = symbols(f'y{i}')
    
    # Create ordered list of all variables
    vars_list = []
    for i in range(num_points):
        vars_list.extend([vars_dict[f'x{i}'], vars_dict[f'y{i}']])
    
    # Calculate derivatives for all variables
    derivatives = []
    symbolic_derivatives = []  # Store symbolic derivatives
    
    for var in vars_list:
        try:
            deriv = diff(gamma_symbolic_function, var).simplify()
        except Exception:
            deriv = 0  # If variable doesn't exist in function, derivative is 0
        symbolic_derivatives.append(deriv)
    
    # Step 3: Evaluate gamma function and its gradients at given coordinates
    # Create substitution dictionary
    subs_dict = {}
    for i in range(num_points):
        if 2*i < len(coords_predicted_scaled):
            subs_dict[vars_dict[f'x{i}']] = coords_predicted_scaled[2*i]
            subs_dict[vars_dict[f'y{i}']] = coords_predicted_scaled[2*i + 1]
    
    # Evaluate function value
    gamma_value = float(gamma_symbolic_function.subs(subs_dict))
    
    # Evaluate all derivatives
    gamma_gradient = np.array([float(deriv.subs(subs_dict)) if not deriv == 0 else 0.0 
                      for deriv in symbolic_derivatives])
    
    return gamma_value, gamma_gradient.reshape(-1, 1), symbolic_derivatives

# Example usage
def main():
    # MATRIX B
    B =   np.array([[ 1, -1,  0,  0,  0,  0,  0,  0,  0],
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
                    [ 0,  0,  0,  1,  0,  0,  0,  0, -1]])
    B_gamma = B[:6, :4]
    
    # Example desired relative positions
    desired_relative_positions = np.array([
        [10, 0.0], 
        [10, 0.0], 
        [-10, 0], 
        [20, 0],
        [15, 5],
        [5, 15]
    ])

    # Example scaled coordinates [x0,y0,x1,y1,x2,y2,x3,y3,,x4,y4,x5,y5,x6,y6,x7,y7]
    coords_predicted_scaled =[0, 0,   # x0, y0
             1, 1,   # x1, y1
             2, 2,   # x2, y2
             3, 3,   # x3, y3
             4, 4,   # x4, y4
             5, 5,   # x5, y5
             6, 6,   # x6, y6
             7, 7,   # x7, y7
             8, 8]   # x8, y8
    
    # Calculate gamma value, gradient, and symbolic derivatives
    gamma_value, gamma_gradient, symbolic_derivatives = gamma_value_and_gradient(B_gamma, B, desired_relative_positions, coords_predicted_scaled)
    
    print("\nΓ value:")
    print(gamma_value)
    
    print("\nΓ gradient:")
    print(gamma_gradient)
    
    print("\nSymbolic derivatives:")
    vars_list = []
    for i in range(B_gamma.shape[1]):
        vars_list.extend([f'x{i}', f'y{i}'])
    for var, deriv in zip(vars_list, symbolic_derivatives):
        print(f"dΓ/d{var} = {deriv}")

if __name__ == "__main__":
    main()