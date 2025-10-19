"""
phase_space_analysis.py
Arctic Permafrost-Carbon-Climate Feedback Model
Phase space analysis and trajectory sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from parameters import PARAMS, SCENARIOS
from solver import run_model
import os


def phase_portrait_multivariate(results, save_path=None):
    """
    Create comprehensive phase portraits for multiple variable pairs
    
    Parameters:
        results : dict
            Model results
        save_path : str, optional
            Path to save figure
    """
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    scenario = results['scenario']
    fig.suptitle(f'Phase Space Analysis: {scenario}', 
                 fontsize=16, fontweight='bold')
    
    # Variable pairs to plot
    pairs = [
        ('C_atm', 'T_s', 'Atmospheric Carbon (Pg C)', 'Temperature (°C)'),
        ('C_deep', 'T_s', 'Deep Permafrost (Pg C)', 'Temperature (°C)'),
        ('C_active', 'T_s', 'Active Layer (Pg C)', 'Temperature (°C)'),
        ('C_atm', 'C_deep', 'Atmospheric C (Pg C)', 'Deep Permafrost (Pg C)'),
        ('CO2_ppm', 'T_s', 'CO₂ (ppm)', 'Temperature (°C)'),
        ('C_active', 'C_deep', 'Active Layer (Pg C)', 'Deep Permafrost (Pg C)')
    ]
    
    for idx, (var1, var2, label1, label2) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        x = results[var1]
        y = results[var2]
        t = results['time']
        
        # Color by time
        scatter = ax.scatter(x, y, c=t, cmap='viridis', s=20, alpha=0.6)
        
        # Mark start and end
        ax.plot(x[0], y[0], 'go', markersize=12, label='Start (2000)', zorder=5)
        ax.plot(x[-1], y[-1], 'ro', markersize=12, label='End', zorder=5)
        
        # Add direction arrows
        n_arrows = 8
        arrow_indices = np.linspace(0, len(x)-10, n_arrows, dtype=int)
        for i in arrow_indices:
            if i < len(x) - 1:
                ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                          arrowprops=dict(arrowstyle='->', color='blue', 
                                        lw=1.5, alpha=0.5))
        
        ax.set_xlabel(label1, fontsize=10)
        ax.set_ylabel(label2, fontsize=10)
        ax.set_title(f'{label1.split("(")[0]} vs {label2.split("(")[0]}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar to first plot
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time (years)', fontsize=9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase portraits saved to: {save_path}")
    
    return fig


def trajectory_sensitivity_analysis(perturbation_size=0.1, n_perturbations=5):
    """
    Analyze sensitivity to initial conditions
    
    Parameters:
        perturbation_size : float
            Fractional perturbation to initial conditions (e.g., 0.1 = 10%)
        n_perturbations : int
            Number of perturbed trajectories to compute
    
    Returns:
        results_list : list
            List of results for each perturbation
    """
    
    print("=" * 70)
    print("TRAJECTORY SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Base initial conditions
    initial_base = np.array([
        PARAMS['C_atm_0'],
        PARAMS['C_active_0'],
        PARAMS['C_deep_0'],
        PARAMS['T_s_0']
    ])
    
    time_points = np.linspace(0, 100, 1001)
    
    print(f"\nBase initial conditions:")
    print(f"  C_atm:    {initial_base[0]:.1f} Pg C")
    print(f"  C_active: {initial_base[1]:.1f} Pg C")
    print(f"  C_deep:   {initial_base[2]:.1f} Pg C")
    print(f"  T_s:      {initial_base[3]:.1f} °C")
    print(f"\nPerturbation: ±{100*perturbation_size:.0f}%")
    print(f"Number of perturbed trajectories: {n_perturbations}")
    
    # Run base case
    print("\nRunning base case...")
    results_base = run_model(initial_base, time_points, PARAMS, 
                            forcing_scenario='RCP4.5')
    
    results_list = [results_base]
    
    # Generate random perturbations
    np.random.seed(42)  # Reproducibility
    
    print(f"\nRunning {n_perturbations} perturbed trajectories...")
    
    for i in range(n_perturbations):
        # Random perturbation
        perturbation = np.random.uniform(-perturbation_size, perturbation_size, 4)
        initial_perturbed = initial_base * (1 + perturbation)
        
        # Ensure positive values
        initial_perturbed = np.maximum(initial_perturbed, 0.1)
        
        results_perturbed = run_model(initial_perturbed, time_points, PARAMS,
                                      forcing_scenario='RCP4.5')
        
        results_list.append(results_perturbed)
        
        if (i + 1) % max(1, n_perturbations // 5) == 0:
            print(f"  Progress: {100*(i+1)//n_perturbations}%")
    
    return results_list


def plot_trajectory_sensitivity(results_list, save_path=None):
    """
    Plot trajectory divergence from initial perturbations
    
    Parameters:
        results_list : list
            List of results (first is base case)
        save_path : str, optional
            Path to save figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Trajectory Sensitivity to Initial Conditions', 
                 fontsize=14, fontweight='bold')
    
    results_base = results_list[0]
    years_base = results_base['time'] + 2000
    
    # Plot base trajectory
    axes[0, 0].plot(years_base, results_base['CO2_ppm'], 'k-', 
                   linewidth=3, label='Base', alpha=0.8)
    axes[0, 1].plot(years_base, results_base['T_s'], 'k-', 
                   linewidth=3, label='Base', alpha=0.8)
    axes[1, 0].plot(years_base, results_base['C_deep'], 'k-', 
                   linewidth=3, label='Base', alpha=0.8)
    
    # Plot perturbed trajectories
    for i, results in enumerate(results_list[1:]):
        years = results['time'] + 2000
        alpha = 0.4
        
        axes[0, 0].plot(years, results['CO2_ppm'], 'b-', 
                       linewidth=1.5, alpha=alpha)
        axes[0, 1].plot(years, results['T_s'], 'r-', 
                       linewidth=1.5, alpha=alpha)
        axes[1, 0].plot(years, results['C_deep'], 'g-', 
                       linewidth=1.5, alpha=alpha)
    
    # Calculate divergence over time
    divergence_co2 = []
    divergence_T = []
    
    for results in results_list[1:]:
        div_co2 = np.abs(results['CO2_ppm'] - results_base['CO2_ppm'])
        div_T = np.abs(results['T_s'] - results_base['T_s'])
        divergence_co2.append(div_co2)
        divergence_T.append(div_T)
    
    # Plot mean divergence
    mean_div_co2 = np.mean(divergence_co2, axis=0)
    std_div_co2 = np.std(divergence_co2, axis=0)
    
    axes[1, 1].plot(years_base, mean_div_co2, 'b-', linewidth=2, 
                   label='CO₂ divergence')
    axes[1, 1].fill_between(years_base, 
                            mean_div_co2 - std_div_co2,
                            mean_div_co2 + std_div_co2,
                            alpha=0.3, color='blue')
    
    mean_div_T = np.mean(divergence_T, axis=0)
    std_div_T = np.std(divergence_T, axis=0)
    
    ax2 = axes[1, 1].twinx()
    ax2.plot(years_base, mean_div_T, 'r-', linewidth=2, 
            label='Temp divergence')
    ax2.fill_between(years_base,
                     mean_div_T - std_div_T,
                     mean_div_T + std_div_T,
                     alpha=0.3, color='red')
    
    # Format axes
    axes[0, 0].set_ylabel('CO₂ (ppm)', fontsize=11)
    axes[0, 0].set_title('Atmospheric CO₂', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0, 1].set_title('Surface Temperature', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Year', fontsize=11)
    axes[1, 0].set_ylabel('Carbon (Pg C)', fontsize=11)
    axes[1, 0].set_title('Deep Permafrost Carbon', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Year', fontsize=11)
    axes[1, 1].set_ylabel('CO₂ Divergence (ppm)', fontsize=11, color='blue')
    axes[1, 1].set_title('Trajectory Divergence', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='y', labelcolor='blue')
    axes[1, 1].grid(True, alpha=0.3)
    
    ax2.set_ylabel('Temperature Divergence (°C)', fontsize=11, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend for divergence plot
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity analysis saved to: {save_path}")
    
    return fig


def analyze_lyapunov_divergence(results_list):
    """
    Calculate approximate Lyapunov exponent from trajectory divergence
    
    Parameters:
        results_list : list
            Results from sensitivity analysis
    
    Returns:
        lyapunov : float
            Approximate Lyapunov exponent
    """
    
    print("\n" + "=" * 70)
    print("LYAPUNOV DIVERGENCE ANALYSIS")
    print("=" * 70)
    
    results_base = results_list[0]
    
    # Calculate divergence in state space
    divergences = []
    
    for results in results_list[1:]:
        # Euclidean distance in normalized state space
        state_base = np.array([results_base['C_atm'], 
                              results_base['C_active'],
                              results_base['C_deep'], 
                              results_base['T_s']])
        
        state_perturbed = np.array([results['C_atm'], 
                                   results['C_active'],
                                   results['C_deep'], 
                                   results['T_s']])
        
        # Normalize by initial values
        state_base_norm = state_base / state_base[:, 0:1]
        state_perturbed_norm = state_perturbed / state_base[:, 0:1]
        
        # Calculate distance
        dist = np.linalg.norm(state_perturbed_norm - state_base_norm, axis=0)
        divergences.append(dist)
    
    # Average divergence
    mean_divergence = np.mean(divergences, axis=0)
    time = results_base['time']
    
    # Fit exponential growth: d(t) = d0 * exp(λ * t)
    # Take log: log(d) = log(d0) + λ * t
    
    # Use middle portion of trajectory (avoid transients)
    mid_start = len(time) // 4
    mid_end = 3 * len(time) // 4
    
    log_div = np.log(mean_divergence[mid_start:mid_end] + 1e-10)
    time_mid = time[mid_start:mid_end]
    
    # Linear fit
    coeffs = np.polyfit(time_mid, log_div, 1)
    lyapunov = coeffs[0]
    
    print(f"\nApproximate Lyapunov exponent: {lyapunov:.6f} yr⁻¹")
    
    if lyapunov > 0:
        doubling_time = np.log(2) / lyapunov
        print(f"Trajectory doubling time: {doubling_time:.1f} years")
        print("Interpretation: Positive λ indicates sensitive dependence on initial conditions")
    elif lyapunov < 0:
        print("Interpretation: Negative λ indicates trajectories converge (stable)")
    else:
        print("Interpretation: λ ≈ 0 indicates neutral stability")
    
    return lyapunov


def main():
    """Main execution"""
    
    os.makedirs('../figures', exist_ok=True)
    
    # 1. Phase portraits for RCP 4.5
    print("\n" + "=" * 70)
    print("CREATING PHASE PORTRAITS")
    print("=" * 70)
    
    initial_state = np.array([PARAMS['C_atm_0'], PARAMS['C_active_0'],
                             PARAMS['C_deep_0'], PARAMS['T_s_0']])
    time_points = np.linspace(0, 100, 1001)
    
    results_45 = run_model(initial_state, time_points, PARAMS, 'RCP4.5')
    
    phase_portrait_multivariate(results_45, 
                                save_path='../figures/phase_portraits_multivariate.png')
    
    # 2. Trajectory sensitivity analysis
    print("\n")
    results_list = trajectory_sensitivity_analysis(perturbation_size=0.1, 
                                                   n_perturbations=10)
    
    plot_trajectory_sensitivity(results_list, 
                               save_path='../figures/trajectory_sensitivity.png')
    
    # 3. Lyapunov analysis
    lyapunov = analyze_lyapunov_divergence(results_list)
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("PHASE SPACE ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results_list


if __name__ == "__main__":
    results = main()