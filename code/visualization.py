"""
visualization.py
Arctic Permafrost-Carbon-Climate Feedback Model
All plotting and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100


def plot_time_series(results_dict, save_path=None):
    """
    Create multi-panel time series plot for multiple scenarios
    
    Parameters:
        results_dict : dict
            Dictionary mapping scenario names to results
        save_path : str, optional
            Path to save figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Permafrost-Carbon-Climate Model Results', 
                 fontsize=14, fontweight='bold')
    
    colors = {'RCP2.6': 'green', 'RCP4.5': 'orange', 'RCP8.5': 'red'}
    
    for scenario, results in results_dict.items():
        years = results['time'] + 2000
        color = colors.get(scenario, 'blue')
        
        # Panel 1: Atmospheric CO2
        axes[0, 0].plot(years, results['CO2_ppm'], 
                       color=color, linewidth=2, label=scenario)
        
        # Panel 2: Surface temperature
        axes[0, 1].plot(years, results['T_s'], 
                       color=color, linewidth=2, label=scenario)
        
        # Panel 3: Active layer carbon
        axes[1, 0].plot(years, results['C_active'], 
                       color=color, linewidth=2, label=scenario)
        
        # Panel 4: Deep permafrost carbon
        axes[1, 1].plot(years, results['C_deep'], 
                       color=color, linewidth=2, label=scenario)
    
    # Format axes
    axes[0, 0].set_ylabel('CO₂ (ppm)')
    axes[0, 0].set_title('Atmospheric CO₂ Concentration')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Arctic Surface Temperature')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Carbon (Pg C)')
    axes[1, 0].set_title('Active Layer Carbon')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Carbon (Pg C)')
    axes[1, 1].set_title('Deep Permafrost Carbon')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_phase_portrait(results, state_vars=('C_atm', 'T_s'), save_path=None):
    """
    Create phase space plot (trajectory in state space)
    
    Parameters:
        results : dict
            Results from run_model()
        state_vars : tuple
            Pair of state variables to plot (e.g., ('C_atm', 'T_s'))
        save_path : str, optional
            Path to save figure
    """
    
    var1_name, var2_name = state_vars
    var1_data = results[var1_name]
    var2_data = results[var2_name]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot trajectory
    ax.plot(var1_data, var2_data, 'b-', linewidth=2, alpha=0.7)
    
    # Mark start and end points
    ax.plot(var1_data[0], var2_data[0], 'go', markersize=10, 
            label='Start (2000)', zorder=5)
    ax.plot(var1_data[-1], var2_data[-1], 'ro', markersize=10, 
            label='End (2100)', zorder=5)
    
    # Add arrows to show direction
    n_arrows = 5
    arrow_indices = np.linspace(0, len(var1_data)-1, n_arrows+1, dtype=int)[:-1]
    for i in arrow_indices[1:]:
        dx = var1_data[i+1] - var1_data[i]
        dy = var2_data[i+1] - var2_data[i]
        ax.arrow(var1_data[i], var2_data[i], dx*0.3, dy*0.3,
                head_width=abs(dx)*0.05, head_length=abs(dx)*0.05,
                fc='blue', ec='blue', alpha=0.6)
    
    # Labels
    labels = {
        'C_atm': 'Atmospheric Carbon (Pg C)',
        'C_active': 'Active Layer Carbon (Pg C)',
        'C_deep': 'Deep Permafrost Carbon (Pg C)',
        'T_s': 'Surface Temperature (°C)',
        'CO2_ppm': 'CO₂ Concentration (ppm)'
    }
    
    ax.set_xlabel(labels.get(var1_name, var1_name))
    ax.set_ylabel(labels.get(var2_name, var2_name))
    ax.set_title(f'Phase Portrait: {results["scenario"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_carbon_budget(results, save_path=None):
    """
    Create stacked area plot showing carbon distribution
    
    Parameters:
        results : dict
            Results from run_model()
        save_path : str, optional
            Path to save figure
    """
    
    years = results['time'] + 2000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked areas
    ax.fill_between(years, 0, results['C_atm'], 
                    alpha=0.7, label='Atmosphere', color='skyblue')
    ax.fill_between(years, results['C_atm'], 
                    results['C_atm'] + results['C_active'],
                    alpha=0.7, label='Active Layer', color='lightgreen')
    ax.fill_between(years, results['C_atm'] + results['C_active'],
                    results['C_atm'] + results['C_active'] + results['C_deep'],
                    alpha=0.7, label='Deep Permafrost', color='brown')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Carbon (Pg C)')
    ax.set_title(f'Carbon Distribution: {results["scenario"]}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_feedback_comparison(results_baseline, results_feedback, save_path=None):
    """
    Compare model runs with and without albedo feedback
    
    Parameters:
        results_baseline : dict
            Results without feedback
        results_feedback : dict
            Results with feedback
        save_path : str, optional
            Path to save figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    years_base = results_baseline['time'] + 2000
    years_feed = results_feedback['time'] + 2000
    
    # Temperature comparison
    axes[0].plot(years_base, results_baseline['T_s'], 'b-', 
                linewidth=2, label='No albedo feedback')
    axes[0].plot(years_feed, results_feedback['T_s'], 'r-', 
                linewidth=2, label='With albedo feedback')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Surface Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CO2 comparison
    axes[1].plot(years_base, results_baseline['CO2_ppm'], 'b-', 
                linewidth=2, label='No albedo feedback')
    axes[1].plot(years_feed, results_feedback['CO2_ppm'], 'r-', 
                linewidth=2, label='With albedo feedback')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('CO₂ (ppm)')
    axes[1].set_title('Atmospheric CO₂')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Albedo Feedback Impact: {results_baseline["scenario"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_sensitivity_analysis(param_name, param_values, results_list, save_path=None):
    """
    Plot results from parameter sensitivity analysis
    
    Parameters:
        param_name : str
            Name of parameter being varied
        param_values : array-like
            Values of parameter
        results_list : list
            List of results dicts for each parameter value
        save_path : str, optional
            Path to save figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Sensitivity Analysis: {param_name}', 
                 fontsize=14, fontweight='bold')
    
    # Extract final values (year 2100)
    final_CO2 = [r['CO2_ppm'][-1] for r in results_list]
    final_T = [r['T_s'][-1] for r in results_list]
    final_C_active = [r['C_active'][-1] for r in results_list]
    final_C_deep = [r['C_deep'][-1] for r in results_list]
    
    # Plot final values vs parameter
    axes[0, 0].plot(param_values, final_CO2, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Final CO₂ (ppm)')
    axes[0, 0].set_title('CO₂ in 2100')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(param_values, final_T, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('Final Temperature (°C)')
    axes[0, 1].set_title('Temperature in 2100')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(param_values, final_C_active, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Active Layer C (Pg C)')
    axes[1, 0].set_title('Active Layer Carbon in 2100')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(param_values, final_C_deep, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Deep Permafrost C (Pg C)')
    axes[1, 1].set_title('Deep Permafrost in 2100')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test visualization functions"""
    print("Visualization module loaded successfully!")
    print("Available plotting functions:")
    print("  - plot_time_series()")
    print("  - plot_phase_portrait()")
    print("  - plot_carbon_budget()")
    print("  - plot_feedback_comparison()")
    print("  - plot_sensitivity_analysis()")