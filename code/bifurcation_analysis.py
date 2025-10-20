"""
phase4_bifurcation_analysis.py
Tasks 4.8-4.10: Bifurcation Analysis and Tipping Points
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.linalg import eig
import sys
import os

sys.path.append(os.path.dirname(__file__))
from parameters import PARAMS
from model import permafrost_model


# =============================================================================
# TASK 4.8: Implement Parameter Sweep
# =============================================================================

def parameter_sweep_emissions(param_range=(0, 20), n_points=100, 
                              integration_time=500):
    """
    Sweep anthropogenic emission rate and find steady states
    
    Parameters:
        param_range: tuple, (min, max) emission rate [Pg C/yr]
        n_points: int, number of parameter values to test
        integration_time: float, time to integrate to reach steady state [years]
    
    Returns:
        results: dict with parameter values and steady states
    """
    
    print(f"\n{'='*70}")
    print("PARAMETER SWEEP: Anthropogenic Emission Rate")
    print(f"{'='*70}")
    print(f"Parameter range: {param_range[0]} - {param_range[1]} Pg C/yr")
    print(f"Number of points: {n_points}")
    print(f"Integration time: {integration_time} years")
    print(f"{'='*70}\n")
    
    # Create parameter array
    emission_values = np.linspace(param_range[0], param_range[1], n_points)
    
    # Storage arrays
    steady_T = np.zeros(n_points)
    steady_CO2 = np.zeros(n_points)
    steady_C_active = np.zeros(n_points)
    steady_C_deep = np.zeros(n_points)
    
    # Initial condition (start from pre-industrial)
    initial_state = np.array([594.0, 174.0, 800.0, -2.0])
    
    print("Running parameter sweep...")
    print_interval = max(1, n_points // 20)  # Print progress every 5%
    
    for i, emission_rate in enumerate(emission_values):
        if i % print_interval == 0:
            print(f"  Progress: {i}/{n_points} ({100*i/n_points:.0f}%)")
        
        # Define constant forcing
        def const_forcing(t):
            return emission_rate
        
        # Time array
        t = np.linspace(0, integration_time, 1000)
        
        # Define RHS
        def rhs(state, time):
            return permafrost_model(state, time, PARAMS, const_forcing, 
                                   use_albedo_feedback=True)
        
        # Integrate to steady state
        solution = odeint(rhs, initial_state, t)
        
        # Extract steady state (final values)
        steady_state = solution[-1, :]
        
        steady_T[i] = steady_state[3]
        steady_CO2[i] = steady_state[0]
        steady_C_active[i] = steady_state[1]
        steady_C_deep[i] = steady_state[2]
        
        # Use current steady state as initial condition for next
        # This helps follow the continuous branch
        initial_state = steady_state
    
    print(f"  Progress: {n_points}/{n_points} (100%)")
    print("\nParameter sweep complete!\n")
    
    results = {
        'emission_values': emission_values,
        'steady_T': steady_T,
        'steady_CO2': steady_CO2,
        'steady_C_active': steady_C_active,
        'steady_C_deep': steady_C_deep
    }
    
    return results


def parameter_sweep_backward(param_range=(20, 0), n_points=100,
                             integration_time=500):
    """
    Sweep parameter backward (high to low) to detect hysteresis
    
    This allows us to follow the upper branch if multiple equilibria exist
    """
    
    print(f"\n{'='*70}")
    print("BACKWARD PARAMETER SWEEP (for hysteresis detection)")
    print(f"{'='*70}")
    print(f"Parameter range: {param_range[0]} → {param_range[1]} Pg C/yr")
    print(f"{'='*70}\n")
    
    # Create parameter array (high to low)
    emission_values = np.linspace(param_range[0], param_range[1], n_points)
    
    # Storage arrays
    steady_T = np.zeros(n_points)
    steady_CO2 = np.zeros(n_points)
    
    # Initial condition (start from warm state)
    initial_state = np.array([1200.0, 250.0, 600.0, 10.0])
    
    print("Running backward sweep...")
    
    for i, emission_rate in enumerate(emission_values):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_points}")
        
        def const_forcing(t):
            return emission_rate
        
        t = np.linspace(0, integration_time, 1000)
        
        def rhs(state, time):
            return permafrost_model(state, time, PARAMS, const_forcing, True)
        
        solution = odeint(rhs, initial_state, t)
        steady_state = solution[-1, :]
        
        steady_T[i] = steady_state[3]
        steady_CO2[i] = steady_state[0]
        
        initial_state = steady_state
    
    print(f"  Progress: {n_points}/{n_points} (100%)\n")
    
    results = {
        'emission_values': emission_values,
        'steady_T': steady_T,
        'steady_CO2': steady_CO2
    }
    
    return results


# =============================================================================
# TASK 4.9: Generate Bifurcation Diagram
# =============================================================================

def plot_bifurcation_diagram(forward_results, backward_results=None,
                             save_figure=True):
    """
    Create publication-quality bifurcation diagram
    
    Parameters:
        forward_results: dict from forward parameter sweep
        backward_results: dict from backward sweep (optional, for hysteresis)
        save_figure: bool, whether to save figure
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # -------------------------------------------------------------------------
    # Panel 1: Temperature Bifurcation Diagram
    # -------------------------------------------------------------------------
    ax = axes[0]
    
    # Forward sweep
    ax.plot(forward_results['emission_values'], 
           forward_results['steady_T'],
           'b-', linewidth=2, label='Forward sweep', zorder=2)
    
    # Backward sweep (if provided)
    if backward_results is not None:
        ax.plot(backward_results['emission_values'],
               backward_results['steady_T'],
               'r--', linewidth=2, label='Backward sweep', zorder=2)
        
        # Detect hysteresis region
        # Find where forward and backward diverge
        # (This is a simplified detection)
        forward_interp = np.interp(backward_results['emission_values'],
                                   forward_results['emission_values'],
                                   forward_results['steady_T'])
        
        difference = np.abs(backward_results['steady_T'] - forward_interp)
        hysteresis_mask = difference > 0.5  # Temperature difference > 0.5°C
        
        if np.any(hysteresis_mask):
            hysteresis_emissions = backward_results['emission_values'][hysteresis_mask]
            ax.axvspan(np.min(hysteresis_emissions), 
                      np.max(hysteresis_emissions),
                      alpha=0.2, color='yellow', 
                      label='Hysteresis region', zorder=1)
    
    # Mark critical threshold
    # Find where dT/dF is maximum (steepest change)
    dT = np.diff(forward_results['steady_T'])
    dF = np.diff(forward_results['emission_values'])
    dT_dF = dT / dF
    
    critical_idx = np.argmax(np.abs(dT_dF))
    critical_emission = forward_results['emission_values'][critical_idx]
    critical_T = forward_results['steady_T'][critical_idx]
    
    ax.plot(critical_emission, critical_T, 'ro', markersize=12,
           markeredgewidth=2, markeredgecolor='darkred',
           label=f'Critical point (~{critical_emission:.1f} Pg C/yr)', zorder=3)
    
    ax.axvline(critical_emission, color='red', linestyle=':', alpha=0.5,
              linewidth=1.5)
    
    ax.set_xlabel('Anthropogenic Emission Rate (Pg C/yr)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Steady-State Temperature (°C)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Bifurcation Diagram: Temperature vs. Forcing',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 2: CO₂ Bifurcation Diagram
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    # Convert to ppm
    forward_CO2_ppm = forward_results['steady_CO2'] / 2.124
    
    ax.plot(forward_results['emission_values'], 
           forward_CO2_ppm,
           'b-', linewidth=2, label='Forward sweep', zorder=2)
    
    if backward_results is not None:
        backward_CO2_ppm = backward_results['steady_CO2'] / 2.124
        ax.plot(backward_results['emission_values'],
               backward_CO2_ppm,
               'r--', linewidth=2, label='Backward sweep', zorder=2)
    
    ax.axvline(critical_emission, color='red', linestyle=':', alpha=0.5,
              linewidth=1.5)
    
    # Mark pre-industrial and current levels
    ax.axhline(280, color='green', linestyle='--', alpha=0.5,
              label='Pre-industrial (280 ppm)')
    ax.axhline(420, color='orange', linestyle='--', alpha=0.5,
              label='~Current (420 ppm)')
    
    ax.set_xlabel('Anthropogenic Emission Rate (Pg C/yr)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Steady-State CO₂ (ppm)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Bifurcation Diagram: Atmospheric CO₂ vs. Forcing',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/bifurcation_diagram.png', dpi=300,
                   bbox_inches='tight')
        print(f"\nFigure saved: ../figures/bifurcation_diagram.png")
    
    return fig


def plot_derivative_diagram(results, save_figure=True):
    """
    Plot dT/dF to identify sensitivity and tipping points
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate derivative
    dT = np.diff(results['steady_T'])
    dF = np.diff(results['emission_values'])
    dT_dF = dT / dF
    
    # Use midpoint emission values
    emission_mid = (results['emission_values'][:-1] + 
                   results['emission_values'][1:]) / 2
    
    ax.plot(emission_mid, dT_dF, 'b-', linewidth=2)
    
    # Mark maximum sensitivity
    max_idx = np.argmax(np.abs(dT_dF))
    ax.plot(emission_mid[max_idx], dT_dF[max_idx], 'ro', markersize=12,
           label=f'Max sensitivity at {emission_mid[max_idx]:.1f} Pg C/yr')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Anthropogenic Emission Rate (Pg C/yr)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Climate Sensitivity (dT/dF) [°C/(Pg C/yr)]',
                 fontsize=12, fontweight='bold')
    ax.set_title('System Sensitivity: Rate of Temperature Change',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/sensitivity_diagram.png', dpi=300,
                   bbox_inches='tight')
        print(f"Figure saved: ../figures/sensitivity_diagram.png")
    
    return fig


# =============================================================================
# TASK 4.10: Identify Critical Thresholds
# =============================================================================

def identify_tipping_points(forward_results, backward_results=None):
    """
    Identify and characterize tipping points
    
    Returns:
        tipping_analysis: dict with tipping point information
    """
    
    print(f"\n{'='*70}")
    print("TIPPING POINT ANALYSIS")
    print(f"{'='*70}\n")
    
    # Method 1: Maximum derivative (steepest change)
    dT = np.diff(forward_results['steady_T'])
    dF = np.diff(forward_results['emission_values'])
    dT_dF = dT / dF
    
    max_deriv_idx = np.argmax(np.abs(dT_dF))
    critical_emission_deriv = forward_results['emission_values'][max_deriv_idx]
    critical_T_deriv = forward_results['steady_T'][max_deriv_idx]
    
    print("Method 1: Maximum Sensitivity Analysis")
    print(f"  Critical emission rate: {critical_emission_deriv:.2f} Pg C/yr")
    print(f"  Temperature at tipping:  {critical_T_deriv:.2f} °C")
    print(f"  Max sensitivity:         {np.max(np.abs(dT_dF)):.3f} °C/(Pg C/yr)")
    print()
    
    # Method 2: Acceleration analysis (second derivative)
    d2T_dF2 = np.diff(dT_dF) / dF[:-1]
    max_accel_idx = np.argmax(np.abs(d2T_dF2))
    critical_emission_accel = forward_results['emission_values'][max_accel_idx]
    
    print("Method 2: Acceleration Analysis")
    print(f"  Critical emission rate: {critical_emission_accel:.2f} Pg C/yr")
    print()
    
    # Method 3: Threshold crossing (e.g., T = 0°C)
    T_threshold = 0.0
    above_threshold = forward_results['steady_T'] > T_threshold
    
    if np.any(above_threshold):
        crossing_idx = np.where(above_threshold)[0][0]
        crossing_emission = forward_results['emission_values'][crossing_idx]
        
        print("Method 3: Temperature Threshold Crossing")
        print(f"  Emission rate at T = {T_threshold}°C: {crossing_emission:.2f} Pg C/yr")
        print()
    
    # Method 4: Hysteresis detection
    if backward_results is not None:
        print("Method 4: Hysteresis Analysis")
        
        # Compare forward and backward sweeps
        # Interpolate to common emission values
        common_emissions = forward_results['emission_values']
        backward_T_interp = np.interp(common_emissions,
                                      backward_results['emission_values'],
                                      backward_results['steady_T'])
        
        difference = np.abs(forward_results['steady_T'] - backward_T_interp)
        
        hysteresis_mask = difference > 0.5  # 0.5°C threshold
        
        if np.any(hysteresis_mask):
            hysteresis_emissions = common_emissions[hysteresis_mask]
            print(f"  Hysteresis detected!")
            print(f"  Lower bound: {np.min(hysteresis_emissions):.2f} Pg C/yr")
            print(f"  Upper bound: {np.max(hysteresis_emissions):.2f} Pg C/yr")
            print(f"  Width: {np.max(hysteresis_emissions) - np.min(hysteresis_emissions):.2f} Pg C/yr")
        else:
            print(f"  No significant hysteresis detected")
        print()
    
    # Summary
    print("="*70)
    print("TIPPING POINT SUMMARY")
    print("="*70)
    print(f"Primary tipping point: {critical_emission_deriv:.2f} Pg C/yr")
    print(f"Temperature at tipping: {critical_T_deriv:.2f} °C")
    print(f"\nCurrent global emissions: ~10 Pg C/yr (for reference)")
    
    if critical_emission_deriv < 10:
        print(f"⚠️  WARNING: Tipping point BELOW current emission rate!")
    else:
        margin = critical_emission_deriv - 10
        print(f"Safety margin: {margin:.1f} Pg C/yr above current rate")
    
    print("="*70 + "\n")
    
    tipping_analysis = {
        'critical_emission': critical_emission_deriv,
        'critical_temperature': critical_T_deriv,
        'max_sensitivity': np.max(np.abs(dT_dF))
    }
    
    return tipping_analysis


# =============================================================================
# COMPREHENSIVE BIFURCATION ANALYSIS
# =============================================================================

def run_complete_bifurcation_analysis():
    """
    Execute full bifurcation analysis pipeline
    """
    
    print(f"\n{'#'*70}")
    print("COMPLETE BIFURCATION ANALYSIS")
    print(f"{'#'*70}\n")
    
    # Task 4.8: Parameter sweep
    print("Step 1: Forward parameter sweep...")
    forward_results = parameter_sweep_emissions(
        param_range=(0, 20),
        n_points=150,
        integration_time=500
    )
    
    print("\nStep 2: Backward parameter sweep (for hysteresis)...")
    backward_results = parameter_sweep_backward(
        param_range=(20, 0),
        n_points=150,
        integration_time=500
    )
    
    # Task 4.9: Generate bifurcation diagrams
    print("\nStep 3: Generating bifurcation diagrams...")
    fig1 = plot_bifurcation_diagram(forward_results, backward_results, 
                                    save_figure=True)
    fig2 = plot_derivative_diagram(forward_results, save_figure=True)
    
    # Task 4.10: Identify tipping points
    print("\nStep 4: Identifying tipping points...")
    tipping_analysis = identify_tipping_points(forward_results, 
                                               backward_results)
    
    # Save results
    os.makedirs('../results', exist_ok=True)
    
    # Save forward sweep
    data_forward = np.column_stack([
        forward_results['emission_values'],
        forward_results['steady_T'],
        forward_results['steady_CO2'],
        forward_results['steady_C_active'],
        forward_results['steady_C_deep']
    ])
    
    header = 'emission_rate,steady_T,steady_CO2,steady_C_active,steady_C_deep'
    np.savetxt('../results/bifurcation_forward.csv', data_forward,
              delimiter=',', header=header, comments='')
    
    print(f"\nResults saved to: ../results/bifurcation_forward.csv")
    
    print(f"\n{'#'*70}")
    print("BIFURCATION ANALYSIS COMPLETE")
    print(f"{'#'*70}\n")
    
    return forward_results, backward_results, tipping_analysis


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute all bifurcation analysis tasks
    """
    
    forward, backward, tipping = run_complete_bifurcation_analysis()
    
    plt.show()
    
    return forward, backward, tipping


if __name__ == "__main__":
    results = main()