"""
Solver functions for running simulations
"""

import numpy as np
from scipy.integrate import odeint
from model import permafrost_model

def run_simulation(initial_conditions, t_span, params, E_anthro, scenario_name=""):
    """
    Run a single simulation
    
    Parameters:
    initial_conditions : array - [C_atm_0, C_active_0, C_deep_0, T_0]
    t_span : array - Time points [years]
    params : dict - Model parameters
    E_anthro : float - Anthropogenic emissions [Pg C/year]
    scenario_name : str - Name for printing
    
    Returns:
    t : array - Time points
    solution : array - State variables at each time point
    """
    print(f"Running simulation: {scenario_name}")
    print(f"  Emissions: {E_anthro} Pg C/year")
    print(f"  Duration: {t_span[-1]} years")
    
    # Solve ODEs
    solution = odeint(
        permafrost_model,
        initial_conditions,
        t_span,
        args=(params, E_anthro),
        rtol=1e-8,
        atol=1e-10
    )
    
    print(f"  Complete! Final state:")
    print(f"    C_atm = {solution[-1, 0]:.1f} Pg C")
    print(f"    C_active = {solution[-1, 1]:.1f} Pg C")
    print(f"    C_deep = {solution[-1, 2]:.1f} Pg C")
    print(f"    T = {solution[-1, 3]:.2f} K ({solution[-1, 3]-273:.2f}°C)")
    print()
    
    return t_span, solution


def run_all_scenarios(initial_conditions, params):
    """
    Run all emission scenarios
    
    Parameters:
    initial_conditions : array - Initial state
    params : dict - Model parameters
    
    Returns:
    results : dict - Results for each scenario
    """
    from parameters import T_FINAL, N_POINTS, E_RCP26, E_BASELINE, E_RCP85
    
    # Time array
    t = np.linspace(0, T_FINAL, N_POINTS)
    
    # Scenarios
    scenarios = {
        'RCP 2.6 (Low Emissions)': E_RCP26,
        'Baseline': E_BASELINE,
        'RCP 8.5 (High Emissions)': E_RCP85
    }
    
    results = {}
    
    for name, emission in scenarios.items():
        t_sim, solution = run_simulation(
            initial_conditions,
            t,
            params,
            emission,
            scenario_name=name
        )
        results[name] = {
            't': t_sim,
            'solution': solution,
            'emission': emission
        }
    
    return results


def save_results(results, filepath):
    """
    Save simulation results to file
    
    Parameters:
    results : dict - Simulation results
    filepath : str - Output file path
    """
    with open(filepath, 'w') as f:
        f.write("Time series results for all scenarios\n")
        f.write("=" * 60 + "\n\n")
        
        for scenario_name, data in results.items():
            f.write(f"\nScenario: {scenario_name}\n")
            f.write(f"Emissions: {data['emission']} Pg C/year\n")
            f.write("-" * 60 + "\n")
            
            solution = data['solution']
            f.write(f"Initial state:\n")
            f.write(f"  C_atm = {solution[0, 0]:.2f} Pg C\n")
            f.write(f"  C_active = {solution[0, 1]:.2f} Pg C\n")
            f.write(f"  C_deep = {solution[0, 2]:.2f} Pg C\n")
            f.write(f"  T = {solution[0, 3]:.2f} K ({solution[0, 3]-273:.2f}°C)\n\n")
            
            f.write(f"Final state:\n")
            f.write(f"  C_atm = {solution[-1, 0]:.2f} Pg C\n")
            f.write(f"  C_active = {solution[-1, 1]:.2f} Pg C\n")
            f.write(f"  C_deep = {solution[-1, 2]:.2f} Pg C\n")
            f.write(f"  T = {solution[-1, 3]:.2f} K ({solution[-1, 3]-273:.2f}°C)\n")
            f.write(f"  ΔT = {solution[-1, 3] - solution[0, 3]:.2f} K\n\n")
    
    print(f"Results saved to: {filepath}")