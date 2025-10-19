"""
solver.py
Arctic Permafrost-Carbon-Climate Feedback Model
ODE solver and simulation runner
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from model import permafrost_model
from parameters import PARAMS, SCENARIOS
from helper_functions import carbon_to_ppm


def run_model(initial_conditions, time_span, params, forcing_scenario='RCP4.5',
              use_albedo_feedback=False):
    """
    Run the permafrost-carbon-climate model
    
    Parameters:
        initial_conditions : array-like, shape (4,)
            Initial state [C_atm, C_active, C_deep, T_s]
        time_span : array-like
            Time points to solve at [years since 2000]
        params : dict
            Model parameters
        forcing_scenario : str
            Emission scenario: 'RCP2.6', 'RCP4.5', or 'RCP8.5'
        use_albedo_feedback : bool
            Whether to include ice-albedo feedback
    
    Returns:
        results : dict containing:
            - 'time': Time array [years]
            - 'C_atm': Atmospheric carbon [Pg C]
            - 'C_active': Active layer carbon [Pg C]
            - 'C_deep': Deep permafrost carbon [Pg C]
            - 'T_s': Surface temperature [°C]
            - 'CO2_ppm': Atmospheric CO2 [ppm]
            - 'scenario': Scenario name
    """
    
    # Get forcing function
    if forcing_scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {forcing_scenario}. "
                        f"Choose from {list(SCENARIOS.keys())}")
    
    forcing_func = SCENARIOS[forcing_scenario]
    
    # Solve ODE system
    solution = odeint(
        permafrost_model,
        initial_conditions,
        time_span,
        args=(params, forcing_func, use_albedo_feedback),
        rtol=1e-6,  # Relative tolerance
        atol=1e-8   # Absolute tolerance
    )
    
    # Extract solution components
    C_atm = solution[:, 0]
    C_active = solution[:, 1]
    C_deep = solution[:, 2]
    T_s = solution[:, 3]
    
    # Convert to CO2 concentration
    CO2_ppm = carbon_to_ppm(C_atm, params)
    
    # Package results
    results = {
        'time': time_span,
        'C_atm': C_atm,
        'C_active': C_active,
        'C_deep': C_deep,
        'T_s': T_s,
        'CO2_ppm': CO2_ppm,
        'scenario': forcing_scenario,
        'albedo_feedback': use_albedo_feedback
    }
    
    return results


def run_multiple_scenarios(initial_conditions, time_span, params,
                           scenarios=None, use_albedo_feedback=False):
    """
    Run model for multiple emission scenarios
    
    Parameters:
        initial_conditions : array-like
            Initial state
        time_span : array-like
            Time points
        params : dict
            Model parameters
        scenarios : list of str, optional
            List of scenarios to run (default: all RCPs)
        use_albedo_feedback : bool
            Ice-albedo feedback switch
    
    Returns:
        results_dict : dict
            Dictionary mapping scenario names to results
    """
    
    if scenarios is None:
        scenarios = ['RCP2.6', 'RCP4.5', 'RCP8.5']
    
    results_dict = {}
    
    for scenario in scenarios:
        print(f"Running {scenario}...")
        results = run_model(
            initial_conditions,
            time_span,
            params,
            forcing_scenario=scenario,
            use_albedo_feedback=use_albedo_feedback
        )
        results_dict[scenario] = results
    
    return results_dict


def save_results(results, filename):
    """
    Save model results to CSV file
    
    Parameters:
        results : dict
            Results from run_model()
        filename : str
            Output filename (e.g., 'results/rcp45_output.csv')
    """
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame({
        'year': results['time'] + 2000,  # Convert to calendar year
        'time_since_2000': results['time'],
        'C_atm_PgC': results['C_atm'],
        'C_active_PgC': results['C_active'],
        'C_deep_PgC': results['C_deep'],
        'T_surface_C': results['T_s'],
        'CO2_ppm': results['CO2_ppm']
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Test solver with a single scenario"""
    
    print("=" * 70)
    print("TESTING PERMAFROST MODEL SOLVER")
    print("=" * 70)
    
    # Set up initial conditions
    initial_state = np.array([
        PARAMS['C_atm_0'],
        PARAMS['C_active_0'],
        PARAMS['C_deep_0'],
        PARAMS['T_s_0']
    ])
    
    print("\nINITIAL CONDITIONS (Year 2000):")
    print(f"  Atmospheric C:     {initial_state[0]:.1f} Pg C")
    print(f"  Active layer C:    {initial_state[1]:.1f} Pg C")
    print(f"  Deep permafrost C: {initial_state[2]:.1f} Pg C")
    print(f"  Surface temp:      {initial_state[3]:.1f} °C")
    
    # Define time span: 2000-2100 (100 years)
    time_points = np.linspace(0, 100, 1001)  # 0.1 year resolution
    
    print(f"\nSIMULATION SETUP:")
    print(f"  Time span:         {time_points[0]:.0f} to {time_points[-1]:.0f} years")
    print(f"  Number of points:  {len(time_points)}")
    print(f"  Time step:         {time_points[1] - time_points[0]:.2f} years")
    
    # Run RCP 4.5 scenario
    print("\n" + "=" * 70)
    print("RUNNING RCP 4.5 SCENARIO...")
    print("=" * 70)
    
    results = run_model(
        initial_state,
        time_points,
        PARAMS,
        forcing_scenario='RCP4.5',
        use_albedo_feedback=False
    )
    
    # Print summary statistics
    print("\nRESULTS SUMMARY:")
    print(f"  Year 2000:")
    print(f"    CO2:         {results['CO2_ppm'][0]:.1f} ppm")
    print(f"    Temperature: {results['T_s'][0]:.2f} °C")
    
    print(f"\n  Year 2100:")
    print(f"    CO2:         {results['CO2_ppm'][-1]:.1f} ppm")
    print(f"    Temperature: {results['T_s'][-1]:.2f} °C")
    
    print(f"\n  Change (2000-2100):")
    print(f"    ΔCO2:        {results['CO2_ppm'][-1] - results['CO2_ppm'][0]:+.1f} ppm")
    print(f"    ΔT:          {results['T_s'][-1] - results['T_s'][0]:+.2f} °C")
    print(f"    ΔC_deep:     {results['C_deep'][-1] - results['C_deep'][0]:+.1f} Pg C")
    
    # Calculate total carbon released from permafrost
    C_perm_initial = initial_state[1] + initial_state[2]
    C_perm_final = results['C_active'][-1] + results['C_deep'][-1]
    C_released = C_perm_initial - C_perm_final
    
    print(f"\n  Permafrost carbon released: {C_released:.1f} Pg C")
    print(f"  Fraction of initial stock:  {100*C_released/C_perm_initial:.1f}%")
    
    # Quick validation checks
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS:")
    print("=" * 70)
    
    # Check 1: CO2 should increase
    co2_increasing = results['CO2_ppm'][-1] > results['CO2_ppm'][0]
    print(f"\n1. CO2 increasing: {'✓ PASS' if co2_increasing else '✗ FAIL'}")
    
    # Check 2: Temperature should increase
    temp_increasing = results['T_s'][-1] > results['T_s'][0]
    print(f"2. Temperature increasing: {'✓ PASS' if temp_increasing else '✗ FAIL'}")
    
    # Check 3: Deep permafrost should decrease
    deep_decreasing = results['C_deep'][-1] < results['C_deep'][0]
    print(f"3. Deep permafrost decreasing: {'✓ PASS' if deep_decreasing else '✗ FAIL'}")
    
    # Check 4: No negative values
    all_positive = (np.all(results['C_atm'] > 0) and 
                   np.all(results['C_active'] >= 0) and 
                   np.all(results['C_deep'] >= 0))
    print(f"4. All carbon stocks positive: {'✓ PASS' if all_positive else '✗ FAIL'}")
    
    # Create quick visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION...")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Permafrost Model Test Run: RCP 4.5', fontsize=14, fontweight='bold')
    
    years = results['time'] + 2000
    
    # Plot 1: CO2 concentration
    axes[0, 0].plot(years, results['CO2_ppm'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('CO₂ (ppm)')
    axes[0, 0].set_title('Atmospheric CO₂')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Surface temperature
    axes[0, 1].plot(years, results['T_s'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Arctic Surface Temperature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Permafrost carbon stocks
    axes[1, 0].plot(years, results['C_active'], 'g-', linewidth=2, label='Active Layer')
    axes[1, 0].plot(years, results['C_deep'], 'brown', linewidth=2, label='Deep Permafrost')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Carbon (Pg C)')
    axes[1, 0].set_title('Permafrost Carbon Stocks')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Total carbon in system
    total_C = results['C_atm'] + results['C_active'] + results['C_deep']
    axes[1, 1].plot(years, total_C, 'k-', linewidth=2)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Total Carbon (Pg C)')
    axes[1, 1].set_title('Total System Carbon')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/test_run_rcp45.png', dpi=150, bbox_inches='tight')
    print("Figure saved to: ../figures/test_run_rcp45.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)