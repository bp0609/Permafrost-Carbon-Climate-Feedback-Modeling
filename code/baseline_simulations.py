"""
phase4_baseline_simulations.py
Tasks 4.1-4.3: Baseline Simulations with Different Forcing Scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os

# Import from existing modules (adjust path if needed)
sys.path.append(os.path.dirname(__file__))
from parameters import PARAMS
from model import permafrost_model
from solver import run_model


# =============================================================================
# TASK 4.1: Define Initial Conditions
# =============================================================================

def get_initial_conditions(scenario='baseline'):
    """
    Define initial conditions for different scenarios
    
    Parameters:
        scenario: str, 'baseline', 'warm_start', or 'cold_start'
    
    Returns:
        initial_state: array [C_atm, C_active, C_deep, T_s]
    """
    if scenario == 'baseline':
        # Pre-industrial conditions (year 2000 baseline)
        C_atm_0 = 594.0      # Atmospheric carbon [Pg C] (~280 ppm)
        C_active_0 = 174.0   # Active layer carbon [Pg C]
        C_deep_0 = 800.0     # Deep permafrost carbon [Pg C]
        T_s_0 = -2.0         # Surface temperature [°C]
        
    elif scenario == 'warm_start':
        # Warmer initial conditions
        C_atm_0 = 700.0
        C_active_0 = 200.0
        C_deep_0 = 750.0
        T_s_0 = 2.0
        
    elif scenario == 'cold_start':
        # Colder initial conditions
        C_atm_0 = 500.0
        C_active_0 = 150.0
        C_deep_0 = 850.0
        T_s_0 = -5.0
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    initial_state = np.array([C_atm_0, C_active_0, C_deep_0, T_s_0])
    
    print(f"\n{'='*70}")
    print(f"INITIAL CONDITIONS: {scenario.upper()}")
    print(f"{'='*70}")
    print(f"Atmospheric CO₂:     {C_atm_0:.1f} Pg C (~{C_atm_0/2.124:.1f} ppm)")
    print(f"Active layer carbon: {C_active_0:.1f} Pg C")
    print(f"Deep permafrost:     {C_deep_0:.1f} Pg C")
    print(f"Surface temperature: {T_s_0:.1f} °C")
    print(f"{'='*70}\n")
    
    return initial_state


# =============================================================================
# TASK 4.2: Implement Forcing Scenarios
# =============================================================================

def rcp_26_emissions(t):
    """
    Low emissions scenario (RCP 2.6 equivalent)
    Peak around 2020, then decline to negative by 2100
    """
    if t < 20:  # 2000-2020: gradual increase
        return 7.0 + 0.3 * t
    elif t < 50:  # 2020-2050: rapid decline
        return 13.0 - 0.4 * (t - 20)
    else:  # 2050-2100: negative emissions
        return 1.0 - 0.03 * (t - 50)


def rcp_45_emissions(t):
    """
    Moderate emissions scenario (RCP 4.5 equivalent)
    Peak around 2040, then stabilize
    """
    if t < 40:  # 2000-2040: increase
        return 7.0 + 0.25 * t
    else:  # 2040-2100: stabilization
        return 17.0 - 0.05 * (t - 40)


def rcp_85_emissions(t):
    """
    High emissions scenario (RCP 8.5 equivalent)
    Continuous increase throughout century
    """
    return 7.0 + 0.35 * t


def pulse_emission(t):
    """
    Abrupt pulse emission (shock test)
    Large pulse at t=20, then return to moderate emissions
    """
    if 20 <= t <= 25:
        return 30.0  # Massive pulse
    else:
        return 7.0   # Background level


def get_forcing_scenarios():
    """
    Return dictionary of all forcing scenarios
    """
    scenarios = {
        'RCP2.6': rcp_26_emissions,
        'RCP4.5': rcp_45_emissions,
        'RCP8.5': rcp_85_emissions,
        'Pulse': pulse_emission
    }
    return scenarios


# =============================================================================
# TASK 4.3: Run Time Series Simulations
# =============================================================================

def run_baseline_simulations(time_span=(0, 200), dt=0.1, save_results=True):
    """
    Run baseline simulations for all forcing scenarios
    
    Parameters:
        time_span: tuple, (t_start, t_end) in years
        dt: float, time step in years
        save_results: bool, whether to save to CSV
    
    Returns:
        results_dict: dictionary of results for each scenario
    """
    
    # Get initial conditions
    initial_state = get_initial_conditions('baseline')
    
    # Time array
    t = np.arange(time_span[0], time_span[1] + dt, dt)
    
    # Get forcing scenarios
    forcing_scenarios = get_forcing_scenarios()
    
    # Storage for results
    results_dict = {}
    
    print(f"\n{'='*70}")
    print("RUNNING BASELINE SIMULATIONS")
    print(f"{'='*70}")
    print(f"Time span: {time_span[0]} - {time_span[1]} years")
    print(f"Time step: {dt} years")
    print(f"Number of scenarios: {len(forcing_scenarios)}")
    print(f"{'='*70}\n")
    
    # Run each scenario
    for scenario_name, forcing_func in forcing_scenarios.items():
        print(f"Running scenario: {scenario_name}...")
        
        # Solve ODE system
        def rhs(state, time):
            return permafrost_model(state, time, PARAMS, forcing_func, 
                                   use_albedo_feedback=True)
        
        solution = odeint(rhs, initial_state, t)
        
        # Store results
        results_dict[scenario_name] = {
            't': t,
            'C_atm': solution[:, 0],
            'C_active': solution[:, 1],
            'C_deep': solution[:, 2],
            'T_s': solution[:, 3],
            'forcing': np.array([forcing_func(ti) for ti in t])
        }
        
        # Print summary statistics
        final_T = solution[-1, 3]
        final_CO2_ppm = solution[-1, 0] / 2.124
        max_T = np.max(solution[:, 3])
        
        print(f"  Final temperature: {final_T:.2f} °C")
        print(f"  Max temperature:   {max_T:.2f} °C")
        print(f"  Final CO₂:         {final_CO2_ppm:.1f} ppm")
        print()
        
        # Save to CSV if requested
        if save_results:
            os.makedirs('../results', exist_ok=True)
            filename = f'../results/baseline_{scenario_name}.csv'
            
            data = np.column_stack([
                t,
                solution[:, 0],
                solution[:, 1],
                solution[:, 2],
                solution[:, 3],
                results_dict[scenario_name]['forcing']
            ])
            
            header = 'time,C_atm,C_active,C_deep,T_s,forcing'
            np.savetxt(filename, data, delimiter=',', header=header, 
                      comments='')
            
            print(f"  Saved to: {filename}")
    
    print(f"\n{'='*70}")
    print("ALL BASELINE SIMULATIONS COMPLETE")
    print(f"{'='*70}\n")
    
    return results_dict


def plot_baseline_results(results_dict, save_figure=True):
    """
    Create comprehensive multi-panel time series plots
    """
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    colors = {
        'RCP2.6': '#2166ac',  # Blue
        'RCP4.5': '#fee08b',  # Yellow
        'RCP8.5': '#d73027',  # Red
        'Pulse': '#8c510a'    # Brown
    }
    
    # Panel 1: Atmospheric CO₂
    ax = axes[0]
    for scenario_name, results in results_dict.items():
        CO2_ppm = results['C_atm'] / 2.124
        ax.plot(results['t'], CO2_ppm, label=scenario_name, 
               color=colors[scenario_name], linewidth=2)
    
    ax.set_ylabel('Atmospheric CO₂ (ppm)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    
    # Panel 2: Surface Temperature
    ax = axes[1]
    for scenario_name, results in results_dict.items():
        ax.plot(results['t'], results['T_s'], label=scenario_name,
               color=colors[scenario_name], linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Surface Temperature (°C)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    
    # Panel 3: Active Layer Carbon
    ax = axes[2]
    for scenario_name, results in results_dict.items():
        ax.plot(results['t'], results['C_active'], label=scenario_name,
               color=colors[scenario_name], linewidth=2)
    
    ax.set_ylabel('Active Layer Carbon (Pg C)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    
    # Panel 4: Deep Permafrost Carbon
    ax = axes[3]
    for scenario_name, results in results_dict.items():
        ax.plot(results['t'], results['C_deep'], label=scenario_name,
               color=colors[scenario_name], linewidth=2)
    
    ax.set_ylabel('Deep Permafrost Carbon (Pg C)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (years from 2000)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/baseline_time_series.png', dpi=300, 
                   bbox_inches='tight')
        print(f"\nFigure saved: ../figures/baseline_time_series.png")
    
    return fig


def create_summary_table(results_dict):
    """
    Create summary table of key metrics for each scenario
    """
    
    print(f"\n{'='*70}")
    print("SUMMARY TABLE: KEY METRICS")
    print(f"{'='*70}")
    print(f"{'Scenario':<12} {'Final T (°C)':<15} {'Max T (°C)':<15} "
          f"{'Final CO₂ (ppm)':<18} {'C Released (Pg)':<15}")
    print(f"{'-'*70}")
    
    for scenario_name, results in results_dict.items():
        final_T = results['T_s'][-1]
        max_T = np.max(results['T_s'])
        final_CO2_ppm = results['C_atm'][-1] / 2.124
        
        # Calculate carbon released from permafrost
        initial_permafrost = results['C_active'][0] + results['C_deep'][0]
        final_permafrost = results['C_active'][-1] + results['C_deep'][-1]
        C_released = initial_permafrost - final_permafrost
        
        print(f"{scenario_name:<12} {final_T:<15.2f} {max_T:<15.2f} "
              f"{final_CO2_ppm:<18.1f} {C_released:<15.1f}")
    
    print(f"{'='*70}\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute all baseline simulation tasks
    """
    
    # Task 4.1 & 4.2: Setup complete (functions defined above)
    
    # Task 4.3: Run simulations
    results_dict = run_baseline_simulations(
        time_span=(0, 200),
        dt=0.1,
        save_results=True
    )
    
    # Create visualizations
    plot_baseline_results(results_dict, save_figure=True)
    
    # Create summary table
    create_summary_table(results_dict)
    
    plt.show()
    
    return results_dict


if __name__ == "__main__":
    results = main()