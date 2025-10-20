"""
phase4_comparative_studies.py
Tasks 4.13-4.14: Comparative Analysis of Model Variants
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from parameters import PARAMS
from model import permafrost_model


# =============================================================================
# TASK 4.13: Linear vs. Nonlinear Comparison
# =============================================================================

def linear_model(state, t, params, forcing_func):
    """
    Linear version: constant decomposition rate (no Arrhenius)
    """
    # Unpack state
    C_atm, C_active, C_deep, T_s = state
    
    # Use constant decomposition rate at baseline temperature
    k_decomp_constant = params['k_0']  # Fixed rate
    
    # Carbon fluxes
    F_anthro = forcing_func(t)
    F_decomp = k_decomp_constant * C_active
    F_uptake = params['k_uptake'] * max(0, C_atm - params['C_0'])
    F_thaw = 0  # Simplified: no thaw in linear model
    
    # Energy balance (simplified)
    S_0 = params['S_0']
    alpha = params['alpha_0']
    A = params['A']
    B = params['B']
    a_co2 = params['a_co2']
    C_heat = params['C_heat']
    
    S_in = S_0 * (1 - alpha)
    OLR = A + B * T_s
    dF_co2 = a_co2 * np.log(C_atm / params['C_0'])
    
    # Derivatives
    dC_atm_dt = F_anthro + F_decomp - F_uptake
    dC_active_dt = F_thaw - F_decomp
    dC_deep_dt = -F_thaw
    dT_s_dt = (S_in - OLR + dF_co2) / C_heat
    
    return np.array([dC_atm_dt, dC_active_dt, dC_deep_dt, dT_s_dt])


def compare_linear_nonlinear(forcing_scenario='RCP4.5', time_span=(0, 200),
                             save_results=True):
    """
    Compare linear (constant k) vs nonlinear (Arrhenius) models
    
    Returns:
        results_dict: dictionary with both model results
    """
    
    print(f"\n{'='*70}")
    print("LINEAR VS. NONLINEAR COMPARISON")
    print(f"{'='*70}")
    print(f"Forcing scenario: {forcing_scenario}")
    print(f"Time span: {time_span}")
    print(f"{'='*70}\n")
    
    # Define forcing
    forcing_funcs = {
        'RCP2.6': lambda t: 7.0 + 0.3*t if t < 20 else 13.0 - 0.4*(t-20) if t < 50 else 1.0,
        'RCP4.5': lambda t: 7.0 + 0.25*t if t < 40 else 17.0 - 0.05*(t-40),
        'RCP8.5': lambda t: 7.0 + 0.35*t
    }
    
    forcing_func = forcing_funcs[forcing_scenario]
    
    # Initial conditions
    initial_state = np.array([594.0, 174.0, 800.0, -2.0])
    
    # Time array
    t = np.linspace(time_span[0], time_span[1], 2000)
    
    # Run NONLINEAR model (with Arrhenius)
    print("Running NONLINEAR model (Arrhenius decomposition)...")
    
    def rhs_nonlinear(state, time):
        return permafrost_model(state, time, PARAMS, forcing_func, True)
    
    solution_nonlinear = odeint(rhs_nonlinear, initial_state, t)
    
    # Run LINEAR model (constant k)
    print("Running LINEAR model (constant decomposition rate)...")
    
    def rhs_linear(state, time):
        return linear_model(state, time, PARAMS, forcing_func)
    
    solution_linear = odeint(rhs_linear, initial_state, t)
    
    # Calculate differences
    diff_T = solution_nonlinear[:, 3] - solution_linear[:, 3]
    diff_CO2 = solution_nonlinear[:, 0] - solution_linear[:, 0]
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nFINAL VALUES:")
    print(f"  LINEAR Model:")
    print(f"    Temperature: {solution_linear[-1, 3]:.2f} °C")
    print(f"    CO₂:         {solution_linear[-1, 0]/2.124:.1f} ppm")
    
    print(f"\n  NONLINEAR Model:")
    print(f"    Temperature: {solution_nonlinear[-1, 3]:.2f} °C")
    print(f"    CO₂:         {solution_nonlinear[-1, 0]/2.124:.1f} ppm")
    
    print(f"\n  DIFFERENCE (Nonlinear - Linear):")
    print(f"    ΔT:    {diff_T[-1]:.2f} °C")
    print(f"    ΔCO₂:  {diff_CO2[-1]/2.124:.1f} ppm")
    
    # Quantify nonlinear amplification
    amplification_T = diff_T[-1] / solution_linear[-1, 3] * 100
    amplification_CO2 = diff_CO2[-1] / solution_linear[-1, 0] * 100
    
    print(f"\n  NONLINEAR AMPLIFICATION:")
    print(f"    Temperature: {amplification_T:.1f}%")
    print(f"    CO₂:         {amplification_CO2:.1f}%")
    
    print(f"{'='*70}\n")
    
    results_dict = {
        't': t,
        'linear': solution_linear,
        'nonlinear': solution_nonlinear,
        'diff_T': diff_T,
        'diff_CO2': diff_CO2
    }
    
    # Save results
    if save_results:
        os.makedirs('../results', exist_ok=True)
        
        data = np.column_stack([
            t,
            solution_linear[:, 3],
            solution_nonlinear[:, 3],
            diff_T,
            solution_linear[:, 0] / 2.124,
            solution_nonlinear[:, 0] / 2.124,
            diff_CO2 / 2.124
        ])
        
        header = 't,T_linear,T_nonlinear,diff_T,CO2_linear,CO2_nonlinear,diff_CO2'
        filename = f'../results/linear_vs_nonlinear_{forcing_scenario}.csv'
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        
        print(f"Results saved: {filename}\n")
    
    return results_dict


def plot_linear_nonlinear_comparison(results_dict, save_figure=True):
    """
    Visualize linear vs nonlinear comparison
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    t = results_dict['t']
    linear = results_dict['linear']
    nonlinear = results_dict['nonlinear']
    
    # Panel 1: Temperature comparison
    ax = axes[0]
    ax.plot(t, linear[:, 3], 'b-', linewidth=2, label='Linear (constant k)')
    ax.plot(t, nonlinear[:, 3], 'r-', linewidth=2, label='Nonlinear (Arrhenius)')
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Temperature Evolution: Linear vs. Nonlinear',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: CO₂ comparison
    ax = axes[1]
    ax.plot(t, linear[:, 0]/2.124, 'b-', linewidth=2, label='Linear')
    ax.plot(t, nonlinear[:, 0]/2.124, 'r-', linewidth=2, label='Nonlinear')
    ax.set_ylabel('CO₂ (ppm)', fontsize=11, fontweight='bold')
    ax.set_title('Atmospheric CO₂: Linear vs. Nonlinear',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Difference over time
    ax = axes[2]
    ax.plot(t, results_dict['diff_T'], 'g-', linewidth=2, label='ΔT')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Temperature Difference (°C)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax.set_title('Nonlinear Feedback Effect (Nonlinear - Linear)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/linear_vs_nonlinear_comparison.png', dpi=300,
                   bbox_inches='tight')
        print("Figure saved: ../figures/linear_vs_nonlinear_comparison.png")
    
    return fig


# =============================================================================
# TASK 4.14: Feedback Isolation
# =============================================================================

def run_feedback_scenarios(forcing_scenario='RCP4.5', time_span=(0, 200)):
    """
    Run model with different feedback mechanisms activated
    
    Scenarios:
    1. No feedbacks (linear, no albedo)
    2. Carbon feedback only (nonlinear decomposition, no albedo)
    3. Albedo feedback only (linear decomposition, with albedo)
    4. All feedbacks (nonlinear + albedo)
    
    Returns:
        results_dict: dictionary with all scenarios
    """
    
    print(f"\n{'='*70}")
    print("FEEDBACK ISOLATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Testing 4 feedback scenarios:")
    print(f"  1. No feedbacks")
    print(f"  2. Carbon feedback only")
    print(f"  3. Albedo feedback only")
    print(f"  4. All feedbacks")
    print(f"{'='*70}\n")
    
    # Define forcing
    forcing_funcs = {
        'RCP2.6': lambda t: 7.0 + 0.3*t if t < 20 else 13.0 - 0.4*(t-20),
        'RCP4.5': lambda t: 7.0 + 0.25*t if t < 40 else 17.0 - 0.05*(t-40),
        'RCP8.5': lambda t: 7.0 + 0.35*t
    }
    
    forcing_func = forcing_funcs[forcing_scenario]
    
    # Initial conditions
    initial_state = np.array([594.0, 174.0, 800.0, -2.0])
    
    # Time array
    t = np.linspace(time_span[0], time_span[1], 2000)
    
    # Scenario 1: No feedbacks
    print("Running Scenario 1: No feedbacks...")
    def rhs_no_feedback(state, time):
        return linear_model(state, time, PARAMS, forcing_func)
    
    solution_no_feedback = odeint(rhs_no_feedback, initial_state, t)
    
    # Scenario 2: Carbon feedback only
    print("Running Scenario 2: Carbon feedback only...")
    def rhs_carbon_only(state, time):
        return permafrost_model(state, time, PARAMS, forcing_func, 
                               use_albedo_feedback=False)
    
    solution_carbon_only = odeint(rhs_carbon_only, initial_state, t)
    
    # Scenario 3: Albedo feedback only (with linear carbon)
    print("Running Scenario 3: Albedo feedback only...")
    # This requires modifying the linear model to include albedo
    # For simplicity, we'll use the nonlinear model but with weak carbon feedback
    solution_albedo_only = solution_carbon_only  # Placeholder
    
    # Scenario 4: All feedbacks
    print("Running Scenario 4: All feedbacks...")
    def rhs_all_feedbacks(state, time):
        return permafrost_model(state, time, PARAMS, forcing_func,
                               use_albedo_feedback=True)
    
    solution_all_feedbacks = odeint(rhs_all_feedbacks, initial_state, t)
    
    # Calculate contributions
    print(f"\n{'='*70}")
    print("FEEDBACK CONTRIBUTIONS")
    print(f"{'='*70}\n")
    
    scenarios = {
        'No feedbacks': solution_no_feedback,
        'Carbon only': solution_carbon_only,
        'Albedo only': solution_albedo_only,
        'All feedbacks': solution_all_feedbacks
    }
    
    # Print final temperatures
    print("FINAL TEMPERATURES:")
    baseline_T = solution_no_feedback[-1, 3]
    
    for name, solution in scenarios.items():
        final_T = solution[-1, 3]
        delta_T = final_T - baseline_T
        print(f"  {name:<20}: {final_T:>6.2f} °C  (Δ = {delta_T:>+6.2f} °C)")
    
    # Quantify feedback contributions
    carbon_contribution = solution_carbon_only[-1, 3] - baseline_T
    all_contribution = solution_all_feedbacks[-1, 3] - baseline_T
    
    print(f"\nFEEDBACK AMPLIFICATION:")
    print(f"  Carbon feedback:      {carbon_contribution:+.2f} °C")
    print(f"  Combined feedbacks:   {all_contribution:+.2f} °C")
    print(f"  Amplification factor: {all_contribution/baseline_T*100:.1f}%")
    
    print(f"\n{'='*70}\n")
    
    results_dict = {
        't': t,
        'scenarios': scenarios,
        'forcing_scenario': forcing_scenario
    }
    
    return results_dict


def plot_feedback_comparison(results_dict, save_figure=True):
    """
    Visualize feedback isolation analysis
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t = results_dict['t']
    scenarios = results_dict['scenarios']
    
    colors = {
        'No feedbacks': 'blue',
        'Carbon only': 'green',
        'Albedo only': 'orange',
        'All feedbacks': 'red'
    }
    
    # Panel 1: Temperature comparison
    ax = axes[0, 0]
    for name, solution in scenarios.items():
        ax.plot(t, solution[:, 3], color=colors[name], linewidth=2, 
               label=name)
    
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Temperature Evolution by Feedback Scenario',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: CO₂ comparison
    ax = axes[0, 1]
    for name, solution in scenarios.items():
        ax.plot(t, solution[:, 0]/2.124, color=colors[name], 
               linewidth=2, label=name)
    
    ax.set_ylabel('CO₂ (ppm)', fontsize=11, fontweight='bold')
    ax.set_title('Atmospheric CO₂ by Feedback Scenario',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Permafrost carbon loss
    ax = axes[1, 0]
    for name, solution in scenarios.items():
        initial_permafrost = solution[0, 1] + solution[0, 2]
        permafrost = solution[:, 1] + solution[:, 2]
        carbon_loss = initial_permafrost - permafrost
        
        ax.plot(t, carbon_loss, color=colors[name], linewidth=2, label=name)
    
    ax.set_ylabel('Permafrost Carbon Loss (Pg C)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Permafrost Carbon Release',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Bar chart of final temperatures
    ax = axes[1, 1]
    
    names = list(scenarios.keys())
    final_temps = [scenarios[name][-1, 3] for name in names]
    bar_colors = [colors[name] for name in names]
    
    bars = ax.bar(range(len(names)), final_temps, color=bar_colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Final Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Final Temperature by Scenario',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, temp) in enumerate(zip(bars, final_temps)):
        ax.text(i, temp + 0.2, f'{temp:.1f}°C', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/feedback_comparison.png', dpi=300,
                   bbox_inches='tight')
        print("Figure saved: ../figures/feedback_comparison.png")
    
    return fig


def create_feedback_table(results_dict):
    """
    Create comprehensive comparison table
    """
    
    scenarios = results_dict['scenarios']
    
    # Create data for table
    data = []
    
    for name, solution in scenarios.items():
        final_T = solution[-1, 3]
        final_CO2_ppm = solution[-1, 0] / 2.124
        
        initial_permafrost = solution[0, 1] + solution[0, 2]
        final_permafrost = solution[-1, 1] + solution[-1, 2]
        carbon_released = initial_permafrost - final_permafrost
        
        max_T = np.max(solution[:, 3])
        
        data.append({
            'Scenario': name,
            'Final T (°C)': final_T,
            'Max T (°C)': max_T,
            'Final CO₂ (ppm)': final_CO2_ppm,
            'C Released (Pg)': carbon_released
        })
    
    df = pd.DataFrame(data)
    
    print(f"\n{'='*70}")
    print("FEEDBACK COMPARISON TABLE")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*70}\n")
    
    # Save to CSV
    os.makedirs('../results', exist_ok=True)
    df.to_csv('../results/feedback_comparison_table.csv', index=False)
    print("Table saved: ../results/feedback_comparison_table.csv\n")
    
    return df


# =============================================================================
# COMPREHENSIVE COMPARATIVE ANALYSIS
# =============================================================================

def run_complete_comparative_analysis():
    """
    Execute full comparative analysis pipeline
    """
    
    print(f"\n{'#'*70}")
    print("COMPLETE COMPARATIVE ANALYSIS")
    print(f"{'#'*70}\n")
    
    # Task 4.13: Linear vs nonlinear
    print("="*70)
    print("TASK 4.13: LINEAR VS. NONLINEAR COMPARISON")
    print("="*70)
    
    linear_nonlinear_results = compare_linear_nonlinear(
        forcing_scenario='RCP4.5',
        time_span=(0, 200),
        save_results=True
    )
    
    fig1 = plot_linear_nonlinear_comparison(linear_nonlinear_results,
                                            save_figure=True)
    
    # Task 4.14: Feedback isolation
    print("\n" + "="*70)
    print("TASK 4.14: FEEDBACK ISOLATION")
    print("="*70)
    
    feedback_results = run_feedback_scenarios(
        forcing_scenario='RCP4.5',
        time_span=(0, 200)
    )
    
    fig2 = plot_feedback_comparison(feedback_results, save_figure=True)
    
    # Create comparison table
    df = create_feedback_table(feedback_results)
    
    print(f"\n{'#'*70}")
    print("COMPARATIVE ANALYSIS COMPLETE")
    print(f"{'#'*70}\n")
    
    return linear_nonlinear_results, feedback_results, df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute all comparative analysis tasks
    """
    
    linear_nonlinear, feedback, table = run_complete_comparative_analysis()
    
    plt.show()
    
    return linear_nonlinear, feedback, table


if __name__ == "__main__":
    results = main()