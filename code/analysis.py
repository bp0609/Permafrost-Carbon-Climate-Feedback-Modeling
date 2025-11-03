"""
Analysis functions: equilibrium, stability, and bifurcation
"""

import numpy as np
from scipy.optimize import fsolve
from model import permafrost_model, jacobian_numerical

def find_equilibrium(initial_guess, params, E_anthro):
    """
    Find equilibrium point (where all derivatives = 0)
    
    Parameters:
    initial_guess : array - Starting guess for equilibrium
    params : dict - Model parameters
    E_anthro : float - Emissions
    
    Returns:
    equilibrium : array - Equilibrium state
    success : bool - Whether solution converged
    """
    def equations(state):
        return permafrost_model(state, 0, params, E_anthro)
    
    solution = fsolve(equations, initial_guess, full_output=True)
    equilibrium = solution[0]
    info = solution[1]
    success = (info['fvec']**2).sum() < 1e-8
    
    return equilibrium, success


def stability_analysis(equilibrium, params, E_anthro):
    """
    Analyze stability of equilibrium point
    
    Parameters:
    equilibrium : array - Equilibrium state
    params : dict - Model parameters
    E_anthro : float - Emissions
    
    Returns:
    eigenvalues : array - Complex eigenvalues
    stability : str - Stability classification
    """
    # Compute Jacobian
    J = jacobian_numerical(equilibrium, params, E_anthro)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    
    # Classify stability
    real_parts = eigenvalues.real
    
    if np.all(real_parts < 0):
        stability = "Stable"
    elif np.all(real_parts > 0):
        stability = "Unstable"
    else:
        stability = "Saddle"
    
    return eigenvalues, stability, J


def bifurcation_analysis(params, E_range, n_points=50):
    """
    Perform bifurcation analysis by varying emissions
    
    Parameters:
    params : dict - Model parameters
    E_range : tuple - (E_min, E_max) emissions range [Pg C/year]
    n_points : int - Number of points
    
    Returns:
    emissions : array - Emission values
    equilibria : dict - Equilibrium states and stability
    """
    from parameters import get_initial_conditions
    
    emissions = np.linspace(E_range[0], E_range[1], n_points)
    
    # Storage
    T_equilibria = []
    C_atm_equilibria = []
    stabilities = []
    
    # Initial guess (start from pre-industrial)
    guess = get_initial_conditions()
    
    print(f"Running bifurcation analysis...")
    print(f"  Emission range: {E_range[0]} - {E_range[1]} Pg C/year")
    print(f"  Number of points: {n_points}")
    
    for i, E in enumerate(emissions):
        # Find equilibrium
        eq, success = find_equilibrium(guess, params, E)
        
        if success:
            # Analyze stability
            eigenvalues, stability, _ = stability_analysis(eq, params, E)
            
            T_equilibria.append(eq[3])
            C_atm_equilibria.append(eq[0])
            stabilities.append(stability)
            
            # Use this equilibrium as next guess
            guess = eq
        else:
            T_equilibria.append(np.nan)
            C_atm_equilibria.append(np.nan)
            stabilities.append("Failed")
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_points}")
    
    print("  Bifurcation analysis complete!")
    print()
    
    return {
        'emissions': emissions,
        'T_equilibria': np.array(T_equilibria),
        'C_atm_equilibria': np.array(C_atm_equilibria),
        'stabilities': stabilities
    }


def analyze_feedbacks(results):
    """
    Quantify feedback contributions
    
    Parameters:
    results : dict - Simulation results
    
    Returns:
    feedback_analysis : dict - Feedback strengths
    """
    feedback_analysis = {}
    
    for scenario_name, data in results.items():
        solution = data['solution']
        
        # Initial and final states
        C_atm_0 = solution[0, 0]
        C_atm_f = solution[-1, 0]
        T_0 = solution[0, 3]
        T_f = solution[-1, 3]
        C_deep_0 = solution[0, 2]
        C_deep_f = solution[-1, 2]
        
        # Changes
        delta_C_atm = C_atm_f - C_atm_0
        delta_T = T_f - T_0
        delta_C_deep = C_deep_f - C_deep_0
        
        # Feedback metrics
        carbon_released = -delta_C_deep  # Carbon lost from deep permafrost
        warming_amplification = delta_T / max(delta_C_atm / C_atm_0, 1e-6)
        
        feedback_analysis[scenario_name] = {
            'delta_T': delta_T,
            'delta_C_atm': delta_C_atm,
            'carbon_released': carbon_released,
            'permafrost_loss_fraction': -delta_C_deep / C_deep_0,
            'warming_amplification': warming_amplification
        }
    
    return feedback_analysis


def save_analysis(equilibria, bifurcation, feedback_analysis, filepath):
    """
    Save analysis results to file
    """
    with open(filepath, 'w') as f:
        f.write("EQUILIBRIUM AND BIFURCATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        # Bifurcation results
        f.write("BIFURCATION ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Emission range: {bifurcation['emissions'][0]:.1f} - {bifurcation['emissions'][-1]:.1f} Pg C/year\n")
        f.write(f"Number of points: {len(bifurcation['emissions'])}\n\n")
        
        # Find tipping points
        stable_mask = np.array(bifurcation['stabilities']) == "Stable"
        if np.any(stable_mask):
            max_stable_T = np.max(bifurcation['T_equilibria'][stable_mask])
            f.write(f"Maximum stable temperature: {max_stable_T:.2f} K ({max_stable_T-273:.2f}°C)\n\n")
        
        # Feedback analysis
        f.write("\nFEEDBACK ANALYSIS\n")
        f.write("-" * 60 + "\n")
        for scenario, data in feedback_analysis.items():
            f.write(f"\n{scenario}:\n")
            f.write(f"  Temperature change: {data['delta_T']:.2f} K\n")
            f.write(f"  Atmospheric carbon change: {data['delta_C_atm']:.1f} Pg C\n")
            f.write(f"  Carbon released from permafrost: {data['carbon_released']:.1f} Pg C\n")
            f.write(f"  Permafrost loss: {data['permafrost_loss_fraction']*100:.1f}%\n")
    
    print(f"Analysis saved to: {filepath}")