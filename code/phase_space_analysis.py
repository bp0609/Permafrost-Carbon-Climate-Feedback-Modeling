"""
phase4_phase_space_analysis.py
Tasks 4.11-4.12: Phase Space and Trajectory Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.dirname(__file__))
from parameters import PARAMS
from model import permafrost_model


# =============================================================================
# TASK 4.11: Create Phase Portraits
# =============================================================================

def phase_portrait_2d(forcing_scenario='RCP4.5', plane=('C_atm', 'T_s'),
                      initial_conditions_list=None, time_span=(0, 200),
                      save_figure=True):
    """
    Create 2D phase portrait in specified plane
    
    Parameters:
        forcing_scenario: str, which emission scenario to use
        plane: tuple, which two variables to plot (x_axis, y_axis)
        initial_conditions_list: list of initial states, or None for default
        time_span: tuple, (t_start, t_end)
        save_figure: bool
    
    Returns:
        fig: matplotlib figure
    """
    
    # Define forcing functions
    forcing_funcs = {
        'RCP2.6': lambda t: 7.0 + 0.3*t if t < 20 else 13.0 - 0.4*(t-20) if t < 50 else 1.0 - 0.03*(t-50),
        'RCP4.5': lambda t: 7.0 + 0.25*t if t < 40 else 17.0 - 0.05*(t-40),
        'RCP8.5': lambda t: 7.0 + 0.35*t
    }
    
    forcing_func = forcing_funcs.get(forcing_scenario, forcing_funcs['RCP4.5'])
    
    # Default initial conditions if not provided
    if initial_conditions_list is None:
        # Vary initial conditions around baseline
        baseline = np.array([594.0, 174.0, 800.0, -2.0])
        
        initial_conditions_list = [
            baseline,
            baseline + np.array([50, 10, -10, 1]),
            baseline + np.array([-50, -10, 10, -1]),
            baseline + np.array([100, 20, -20, 2]),
            baseline + np.array([-100, -20, 20, -2]),
            baseline + np.array([0, 30, -30, 0.5]),
        ]
    
    # Map variable names to indices
    var_map = {'C_atm': 0, 'C_active': 1, 'C_deep': 2, 'T_s': 3}
    idx_x = var_map[plane[0]]
    idx_y = var_map[plane[1]]
    
    # Units and labels
    labels = {
        'C_atm': 'Atmospheric Carbon (Pg C)',
        'C_active': 'Active Layer Carbon (Pg C)',
        'C_deep': 'Deep Permafrost Carbon (Pg C)',
        'T_s': 'Surface Temperature (°C)'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Time array
    t = np.linspace(time_span[0], time_span[1], 2000)
    
    # Define RHS
    def rhs(state, time):
        return permafrost_model(state, time, PARAMS, forcing_func, True)
    
    # Plot trajectories from different initial conditions
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions_list)))
    
    for i, ic in enumerate(initial_conditions_list):
        # Integrate
        solution = odeint(rhs, ic, t)
        
        # Extract x and y components
        x = solution[:, idx_x]
        y = solution[:, idx_y]
        
        # Plot trajectory
        ax.plot(x, y, color=colors[i], linewidth=1.5, alpha=0.7,
               label=f'IC {i+1}')
        
        # Mark initial point
        ax.plot(x[0], y[0], 'o', color=colors[i], markersize=8,
               markeredgewidth=1.5, markeredgecolor='white')
        
        # Mark final point
        ax.plot(x[-1], y[-1], 's', color=colors[i], markersize=8,
               markeredgewidth=1.5, markeredgecolor='white')
        
        # Add direction arrows
        # Show arrows at 25%, 50%, 75% of trajectory
        for frac in [0.25, 0.5, 0.75]:
            idx = int(frac * len(x))
            if idx < len(x) - 10:
                dx = x[idx+5] - x[idx]
                dy = y[idx+5] - y[idx]
                ax.arrow(x[idx], y[idx], dx, dy, 
                        head_width=abs(dx)*0.3, head_length=abs(dy)*0.3,
                        fc=colors[i], ec=colors[i], alpha=0.6)
    
    ax.set_xlabel(labels[plane[0]], fontsize=12, fontweight='bold')
    ax.set_ylabel(labels[plane[1]], fontsize=12, fontweight='bold')
    ax.set_title(f'Phase Portrait: {plane[1]} vs {plane[0]} ({forcing_scenario})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        filename = f'../figures/phase_portrait_{plane[0]}_{plane[1]}_{forcing_scenario}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def phase_portrait_3d(forcing_scenario='RCP4.5',
                      variables=('C_atm', 'T_s', 'C_deep'),
                      initial_conditions_list=None,
                      time_span=(0, 200),
                      save_figure=True):
    """
    Create 3D phase portrait
    
    Parameters:
        forcing_scenario: str
        variables: tuple of 3 variable names
        initial_conditions_list: list of initial states
        time_span: tuple
        save_figure: bool
    """
    
    # Define forcing
    forcing_funcs = {
        'RCP2.6': lambda t: 7.0 + 0.3*t if t < 20 else 13.0 - 0.4*(t-20),
        'RCP4.5': lambda t: 7.0 + 0.25*t if t < 40 else 17.0 - 0.05*(t-40),
        'RCP8.5': lambda t: 7.0 + 0.35*t
    }
    
    forcing_func = forcing_funcs.get(forcing_scenario, forcing_funcs['RCP4.5'])
    
    # Default initial conditions
    if initial_conditions_list is None:
        baseline = np.array([594.0, 174.0, 800.0, -2.0])
        initial_conditions_list = [
            baseline,
            baseline + np.array([100, 20, -20, 2]),
            baseline + np.array([-100, -20, 20, -2]),
        ]
    
    # Map variables to indices
    var_map = {'C_atm': 0, 'C_active': 1, 'C_deep': 2, 'T_s': 3}
    idx = [var_map[v] for v in variables]
    
    # Labels
    labels = {
        'C_atm': 'Atmospheric C (Pg)',
        'C_active': 'Active Layer C (Pg)',
        'C_deep': 'Deep Permafrost C (Pg)',
        'T_s': 'Temperature (°C)'
    }
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time array
    t = np.linspace(time_span[0], time_span[1], 2000)
    
    def rhs(state, time):
        return permafrost_model(state, time, PARAMS, forcing_func, True)
    
    # Plot trajectories
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, ic in enumerate(initial_conditions_list):
        solution = odeint(rhs, ic, t)
        
        x = solution[:, idx[0]]
        y = solution[:, idx[1]]
        z = solution[:, idx[2]]
        
        # Plot trajectory
        ax.plot(x, y, z, color=colors[i % len(colors)], 
               linewidth=2, alpha=0.8, label=f'Trajectory {i+1}')
        
        # Mark start and end
        ax.scatter(x[0], y[0], z[0], color=colors[i % len(colors)], 
                  s=100, marker='o', edgecolors='white', linewidths=2)
        ax.scatter(x[-1], y[-1], z[-1], color=colors[i % len(colors)],
                  s=100, marker='s', edgecolors='white', linewidths=2)
    
    ax.set_xlabel(labels[variables[0]], fontsize=11, fontweight='bold')
    ax.set_ylabel(labels[variables[1]], fontsize=11, fontweight='bold')
    ax.set_zlabel(labels[variables[2]], fontsize=11, fontweight='bold')
    ax.set_title(f'3D Phase Portrait ({forcing_scenario})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        filename = f'../figures/phase_portrait_3d_{forcing_scenario}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def plot_nullclines(forcing_value=7.0, plane=('C_atm', 'T_s'),
                   save_figure=True):
    """
    Plot nullclines (curves where dX/dt = 0 or dY/dt = 0)
    
    This helps visualize where the system is in equilibrium
    """
    
    print(f"\n{'='*70}")
    print(f"COMPUTING NULLCLINES")
    print(f"{'='*70}\n")
    
    # Map variables
    var_map = {'C_atm': 0, 'C_active': 1, 'C_deep': 2, 'T_s': 3}
    idx_x = var_map[plane[0]]
    idx_y = var_map[plane[1]]
    
    # Define forcing
    def const_forcing(t):
        return forcing_value
    
    # Create meshgrid
    if plane[0] == 'C_atm':
        x_range = np.linspace(400, 1200, 40)
    else:
        x_range = np.linspace(50, 300, 40)
    
    if plane[1] == 'T_s':
        y_range = np.linspace(-10, 15, 40)
    else:
        y_range = np.linspace(50, 300, 40)
    
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute derivatives at each point
    dX_dt = np.zeros_like(X)
    dY_dt = np.zeros_like(Y)
    
    baseline_state = np.array([594.0, 174.0, 800.0, -2.0])
    
    print("Computing derivatives on grid...")
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = baseline_state.copy()
            state[idx_x] = X[i, j]
            state[idx_y] = Y[i, j]
            
            derivs = permafrost_model(state, 0, PARAMS, const_forcing, True)
            
            dX_dt[i, j] = derivs[idx_x]
            dY_dt[i, j] = derivs[idx_y]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot nullclines
    ax.contour(X, Y, dX_dt, levels=[0], colors='blue', linewidths=2,
              linestyles='-', label=f'd{plane[0]}/dt = 0')
    ax.contour(X, Y, dY_dt, levels=[0], colors='red', linewidths=2,
              linestyles='--', label=f'd{plane[1]}/dt = 0')
    
    # Add vector field
    skip = 3  # Plot every 3rd arrow
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
             dX_dt[::skip, ::skip], dY_dt[::skip, ::skip],
             alpha=0.4, scale=500)
    
    # Labels
    labels = {
        'C_atm': 'Atmospheric Carbon (Pg C)',
        'C_active': 'Active Layer Carbon (Pg C)',
        'C_deep': 'Deep Permafrost Carbon (Pg C)',
        'T_s': 'Surface Temperature (°C)'
    }
    
    ax.set_xlabel(labels[plane[0]], fontsize=12, fontweight='bold')
    ax.set_ylabel(labels[plane[1]], fontsize=12, fontweight='bold')
    ax.set_title(f'Nullclines and Vector Field (F={forcing_value} Pg C/yr)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        filename = f'../figures/nullclines_{plane[0]}_{plane[1]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    print(f"\n{'='*70}\n")
    
    return fig


# =============================================================================
# TASK 4.12: Trajectory Sensitivity Analysis
# =============================================================================

def trajectory_sensitivity(baseline_state, perturbation_size=0.01,
                          forcing_scenario='RCP4.5',
                          time_span=(0, 200),
                          n_perturbations=10):
    """
    Analyze sensitivity to initial conditions
    
    Parameters:
        baseline_state: array, baseline initial condition
        perturbation_size: float, relative perturbation magnitude
        forcing_scenario: str
        time_span: tuple
        n_perturbations: int, number of perturbed trajectories
    
    Returns:
        results: dict with trajectory data
    """
    
    print(f"\n{'='*70}")
    print("TRAJECTORY SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Baseline state: {baseline_state}")
    print(f"Perturbation size: {perturbation_size*100}%")
    print(f"Number of perturbations: {n_perturbations}")
    print(f"{'='*70}\n")
    
    # Define forcing
    forcing_funcs = {
        'RCP2.6': lambda t: 7.0 + 0.3*t if t < 20 else 13.0 - 0.4*(t-20),
        'RCP4.5': lambda t: 7.0 + 0.25*t if t < 40 else 17.0 - 0.05*(t-40),
        'RCP8.5': lambda t: 7.0 + 0.35*t
    }
    
    forcing_func = forcing_funcs[forcing_scenario]
    
    # Time array
    t = np.linspace(time_span[0], time_span[1], 2000)
    
    def rhs(state, time):
        return permafrost_model(state, time, PARAMS, forcing_func, True)
    
    # Baseline trajectory
    solution_baseline = odeint(rhs, baseline_state, t)
    
    # Perturbed trajectories
    solutions_perturbed = []
    
    print("Running perturbed simulations...")
    
    for i in range(n_perturbations):
        # Random perturbation
        perturbation = np.random.randn(4) * perturbation_size * baseline_state
        perturbed_state = baseline_state + perturbation
        
        # Ensure positive values
        perturbed_state = np.abs(perturbed_state)
        
        solution = odeint(rhs, perturbed_state, t)
        solutions_perturbed.append(solution)
    
    # Calculate divergence metrics
    divergences = []
    
    for solution in solutions_perturbed:
        # Euclidean distance from baseline
        diff = solution - solution_baseline
        distance = np.sqrt(np.sum(diff**2, axis=1))
        divergences.append(distance)
    
    divergences = np.array(divergences)
    
    # Calculate maximum Lyapunov exponent (rough estimate)
    # λ ≈ (1/t) * log(d(t)/d(0))
    d_0 = np.mean([np.linalg.norm(solutions_perturbed[i][0] - baseline_state)
                   for i in range(n_perturbations)])
    
    d_final = np.mean([np.linalg.norm(solutions_perturbed[i][-1] - solution_baseline[-1])
                      for i in range(n_perturbations)])
    
    lyapunov_estimate = np.log(d_final / d_0) / time_span[1]
    
    print(f"\nLyapunov exponent estimate: {lyapunov_estimate:.6f} per year")
    
    if lyapunov_estimate > 0:
        print("  → Positive: Chaotic/sensitive behavior")
    elif lyapunov_estimate < 0:
        print("  → Negative: Stable/converging trajectories")
    else:
        print("  → Zero: Neutral stability")
    
    print(f"\n{'='*70}\n")
    
    results = {
        't': t,
        'baseline': solution_baseline,
        'perturbed': solutions_perturbed,
        'divergences': divergences,
        'lyapunov': lyapunov_estimate
    }
    
    return results


def plot_trajectory_sensitivity(sensitivity_results, save_figure=True):
    """
    Visualize trajectory sensitivity analysis
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t = sensitivity_results['t']
    baseline = sensitivity_results['baseline']
    perturbed_list = sensitivity_results['perturbed']
    divergences = sensitivity_results['divergences']
    
    # Panel 1: Temperature trajectories
    ax = axes[0, 0]
    ax.plot(t, baseline[:, 3], 'k-', linewidth=3, label='Baseline', zorder=3)
    
    for i, solution in enumerate(perturbed_list):
        ax.plot(t, solution[:, 3], alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Temperature Trajectories', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: CO2 trajectories
    ax = axes[0, 1]
    ax.plot(t, baseline[:, 0]/2.124, 'k-', linewidth=3, label='Baseline', zorder=3)
    
    for solution in perturbed_list:
        ax.plot(t, solution[:, 0]/2.124, alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('CO₂ (ppm)', fontsize=11, fontweight='bold')
    ax.set_title('CO₂ Trajectories', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Divergence over time
    ax = axes[1, 0]
    
    for i, div in enumerate(divergences):
        ax.semilogy(t, div, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Divergence from Baseline (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Trajectory Divergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Phase portrait with perturbations
    ax = axes[1, 1]
    ax.plot(baseline[:, 0], baseline[:, 3], 'k-', linewidth=3, 
           label='Baseline', zorder=3)
    
    for solution in perturbed_list:
        ax.plot(solution[:, 0], solution[:, 3], alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Atmospheric Carbon (Pg C)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Phase Portrait: Sensitivity', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/trajectory_sensitivity.png', dpi=300,
                   bbox_inches='tight')
        print(f"Figure saved: ../figures/trajectory_sensitivity.png")
    
    return fig


# =============================================================================
# COMPREHENSIVE PHASE SPACE ANALYSIS
# =============================================================================

def run_complete_phase_space_analysis():
    """
    Execute full phase space analysis pipeline
    """
    
    print(f"\n{'#'*70}")
    print("COMPLETE PHASE SPACE ANALYSIS")
    print(f"{'#'*70}\n")
    
    # Task 4.11: Create phase portraits
    print("Step 1: Creating 2D phase portraits...")
    
    fig1 = phase_portrait_2d(forcing_scenario='RCP4.5',
                            plane=('C_atm', 'T_s'),
                            save_figure=True)
    
    fig2 = phase_portrait_2d(forcing_scenario='RCP4.5',
                            plane=('C_deep', 'T_s'),
                            save_figure=True)
    
    print("\nStep 2: Creating 3D phase portrait...")
    fig3 = phase_portrait_3d(forcing_scenario='RCP4.5',
                            variables=('C_atm', 'T_s', 'C_deep'),
                            save_figure=True)
    
    print("\nStep 3: Computing nullclines...")
    fig4 = plot_nullclines(forcing_value=7.0,
                          plane=('C_atm', 'T_s'),
                          save_figure=True)
    
    # Task 4.12: Trajectory sensitivity
    print("\nStep 4: Trajectory sensitivity analysis...")
    baseline_state = np.array([594.0, 174.0, 800.0, -2.0])
    
    sensitivity_results = trajectory_sensitivity(
        baseline_state=baseline_state,
        perturbation_size=0.05,
        forcing_scenario='RCP4.5',
        time_span=(0, 200),
        n_perturbations=20
    )
    
    fig5 = plot_trajectory_sensitivity(sensitivity_results, save_figure=True)
    
    print(f"\n{'#'*70}")
    print("PHASE SPACE ANALYSIS COMPLETE")
    print(f"{'#'*70}\n")
    
    return sensitivity_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute all phase space analysis tasks
    """
    
    results = run_complete_phase_space_analysis()
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()