import numpy as np
from sympy import symbols, init_printing

def print_symbolic_formulas():
    """Print the symbolic representations of Phi and its gradient"""
    init_printing(use_unicode=True)
    
    Gamma, Beta = symbols('Gamma Beta')
    grad_Gamma, grad_Beta = symbols('∇Gamma ∇Beta')
    k = symbols('k')
    
    Phi = Gamma / (Beta**(1/k))
    # print("\nSymbolic representation of Phi:")
    # print("Φ = Γ / β^(1/k)")
    # print(f"  = {Phi}")
    
    grad_Phi = Beta**(1/k) * grad_Gamma + Gamma * grad_Beta
    # print("\nSymbolic representation of gradient of Phi:")
    # print("∇Φ = β^(1/k) * ∇Γ + Γ * ∇β")
    # print(f"   = {grad_Phi}")
    # print("\n")

def phi_and_its_gradient(gamma_value, beta_value, grad_gamma, grad_beta, k_Beta):
    """
    Calculate Phi and its gradient for a navigational function.
    
    Parameters:
    -----------
    gamma_value : float
        Value of the gamma function
    beta_value : float
        Value of the beta function
    grad_gamma : ndarray
        Gradient of gamma function
    grad_beta : ndarray
        Gradient of beta function
    k : float
        Power coefficient for beta
        
    Returns:
    --------
    phi_value : float
        Calculated value of phi
    phi_gradient : ndarray
        Gradient of phi
    """
    # Ensure inputs are numpy arrays with correct shape
    grad_gamma = np.asarray(grad_gamma).reshape(-1, 1)
    grad_beta = np.asarray(grad_beta).reshape(-1, 1)

    # Calculate the power term
    beta_power = beta_value ** (1/k_Beta)
    
    # Calculate phi value
    phi_value = gamma_value / beta_power
    
    # Calculate phi gradient using the chain rule
    # ∇Φ = β^(1/k) * ∇Γ - Γ * ∇β # Original Min Distance Function 
    phi_gradient = beta_power * grad_gamma - gamma_value * grad_beta

    # Print detailed breakdown of calculations
    # print("\nCalculation Details:")
    # print(f"Input Values:")
    # print(f"Γ (gamma) = {gamma_value}")
    # print(f"β (beta) = {beta_value}")
    # print(f"k = {k_Beta}")
    # print(f"β^(1/k) = {beta_power}")
    
    # print(f"\nΦ (phi) = Γ/β^(1/k) = {phi_value}")
    
    # print("\nGradient Components:")
    # print("∇Φ = β^(1/k) * ∇Γ - Γ * ∇β")
    for i in range(len(phi_gradient)//2):
        # Properly extract single elements from arrays
        x_component = phi_gradient[2*i, 0]
        y_component = phi_gradient[2*i+1, 0]
        grad_gamma_x = grad_gamma[2*i, 0]
        grad_gamma_y = grad_gamma[2*i+1, 0]
        grad_beta_x = grad_beta[2*i, 0]
        grad_beta_y = grad_beta[2*i+1, 0]
        
        # print(f"\nPoint {i}:")
        # print(f"dΦ/dx{i} = {beta_power:.6f} * ({grad_gamma_x:.6f}) - {gamma_value:.6f} * ({grad_beta_x:.6f}) = {x_component:.6f}")
        # print(f"dΦ/dy{i} = {beta_power:.6f} * ({grad_gamma_y:.6f}) - {gamma_value:.6f} * ({grad_beta_y:.6f}) = {y_component:.6f}")

    return phi_value, phi_gradient

def main():
    gamma_val = 15600.0
    beta_val = 0.25773602000919954
    k = 2.0
    Beta_function_used_approximation = True
    
    grad_gamma = np.array([[-40.], [-240.], [-200.], [-80.], [280.], [80.], 
                          [-40.], [240.], [0.], [0.], [0.], [0.], [0.], 
                          [0.], [0.], [0.], [0.], [0.]])
    
    grad_beta = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], 
                         [0.7683588660302745], [-1.438851695937615], [0.0], 
                         [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], 
                         [-0.7683588660302745], [1.438851695937615]])
    
    print_symbolic_formulas()
    
    print("Numerical Calculations:")
    print("-" * 50)
    phi_val, phi_grad = phi_and_its_gradient(gamma_val, beta_val, grad_gamma, grad_beta, k, Beta_function_used_approximation)

if __name__ == "__main__":
    main()