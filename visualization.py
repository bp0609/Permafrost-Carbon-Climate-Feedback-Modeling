"""
Visualization Functions for Arctic Permafrost-Carbon-Climate Model

Functions for creating publication-quality plots:
- Time series of state variables
- Phase space portraits
- Parameter sensitivity analysis
- Feedback comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import model
import parameters as params


# ============================================================================
# PLOTTING STYLE CONFIGURATION
# ============================================================================

def set_plot_style():
    """Set consistent plotting style for all figures"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['grid.alpha'] = 0.3


# ============================================================================
# TIME SERIES PLOTS
# ============================================================================

def plot_time_series(t, solution, title="Model Time Series", filename=None):
    """
    Plot all four state variables over time
    
    Parameters:
    -----------
    t : array
        Time points [years]
    solution : array
        Solution array with shape (len(t), 4)
    title : str
        Plot title
    filename : str, optional
        If provided, save figure to this file
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract state variables
    C_frozen = solution[:, 0]
    C_active = solution[:, 1]
    C_atm = solution[:, 2]
    T_s = solution[:, 3]
    
    # Plot 1: Frozen Permafrost Carbon
    axes[0, 0].plot(t, C_frozen, 'b-', linewidth=2.5)
    axes[0, 0].set_xlabel('Time [years]')
    axes[0, 0].set_ylabel('Frozen Carbon [PgC]')
    axes[0, 0].set_title('Frozen Permafrost Carbon Pool')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([t[0], t[-1]])
    
    # Plot 2: Active Layer Carbon
    axes[0, 1].plot(t, C_active, 'g-', linewidth=2.5)
    axes[0, 1].set_xlabel('Time [years]')
    axes[0, 1].set_ylabel('Active Carbon [PgC]')
    axes[0, 1].set_title('Active Layer Carbon Pool')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([t[0], t[-1]])
    
    # Plot 3: Atmospheric CO2
    axes[1, 0].plot(t, C_atm, 'r-', linewidth=2.5)
    axes[1, 0].set_xlabel('Time [years]')
    axes[1, 0].set_ylabel('Atmospheric CO₂ [PgC]')
    axes[1, 0].set_title('Atmospheric Carbon')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([t[0], t[-1]])
    # Add horizontal line for pre-industrial level
    axes[1, 0].axhline(y=600, color='gray', linestyle='--', 
                       label='Pre-industrial', alpha=0.5)
    axes[1, 0].legend()
    
    # Plot 4: Surface Temperature
    axes[1, 1].plot(t, T_s, 'orange', linewidth=2.5)
    axes[1, 1].set_xlabel('Time [years]')
    axes[1, 1].set_ylabel('Temperature [°C]')
    axes[1, 1].set_title('Surface Temperature')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([t[0], t[-1]])
    # Add horizontal line for reference
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', 
                       label='0°C reference', alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def plot_carbon_pools_stacked(t, solution, title="Carbon Distribution", filename=None):
    """
    Plot carbon pools as stacked area chart
    
    Parameters:
    -----------
    t : array
        Time points
    solution : array
        Solution array
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    C_frozen = solution[:, 0]
    C_active = solution[:, 1]
    C_atm = solution[:, 2]
    
    # Stacked area plot
    ax.fill_between(t, 0, C_frozen, alpha=0.6, label='Frozen Permafrost', color='blue')
    ax.fill_between(t, C_frozen, C_frozen + C_active, alpha=0.6, 
                    label='Active Layer', color='green')
    ax.fill_between(t, C_frozen + C_active, C_frozen + C_active + C_atm, 
                    alpha=0.6, label='Atmosphere', color='red')
    
    # Total carbon line
    C_total = C_frozen + C_active + C_atm
    ax.plot(t, C_total, 'k--', linewidth=2, label='Total Carbon', alpha=0.7)
    
    ax.set_xlabel('Time [years]', fontsize=12)
    ax.set_ylabel('Carbon [PgC]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t[0], t[-1]])
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def plot_comparison_scenarios(results_dict, title="Scenario Comparison", filename=None):
    """
    Compare multiple scenarios on the same plot
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary where keys are scenario names and values are (t, solution) tuples
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (scenario_name, (t, solution)) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        C_frozen = solution[:, 0]
        C_active = solution[:, 1]
        C_atm = solution[:, 2]
        T_s = solution[:, 3]
        
        # Plot each variable
        axes[0, 0].plot(t, C_frozen, color=color, linestyle=linestyle, 
                       label=scenario_name, linewidth=2)
        axes[0, 1].plot(t, C_active, color=color, linestyle=linestyle, 
                       label=scenario_name, linewidth=2)
        axes[1, 0].plot(t, C_atm, color=color, linestyle=linestyle, 
                       label=scenario_name, linewidth=2)
        axes[1, 1].plot(t, T_s, color=color, linestyle=linestyle, 
                       label=scenario_name, linewidth=2)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Time [years]')
    axes[0, 0].set_ylabel('Frozen Carbon [PgC]')
    axes[0, 0].set_title('Frozen Permafrost Carbon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time [years]')
    axes[0, 1].set_ylabel('Active Carbon [PgC]')
    axes[0, 1].set_title('Active Layer Carbon')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time [years]')
    axes[1, 0].set_ylabel('Atmospheric CO₂ [PgC]')
    axes[1, 0].set_title('Atmospheric Carbon')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time [years]')
    axes[1, 1].set_ylabel('Temperature [°C]')
    axes[1, 1].set_title('Surface Temperature')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


# ============================================================================
# PHASE SPACE PLOTS
# ============================================================================

def plot_phase_portrait_2D(solution, var1_idx=2, var2_idx=3, 
                          var1_name='C_atm', var2_name='T_s',
                          var1_label='Atmospheric CO₂ [PgC]', 
                          var2_label='Temperature [°C]',
                          title="Phase Portrait", filename=None):
    """
    Create 2D phase portrait
    
    Parameters:
    -----------
    solution : array
        Solution array
    var1_idx : int
        Index of first variable (default: 2 for C_atm)
    var2_idx : int
        Index of second variable (default: 3 for T_s)
    var1_name : str
        Name of first variable
    var2_name : str
        Name of second variable
    var1_label : str
        Axis label for first variable
    var2_label : str
        Axis label for second variable
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    var1 = solution[:, var1_idx]
    var2 = solution[:, var2_idx]
    
    # Plot trajectory with color gradient
    points = np.array([var1, var2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(np.linspace(0, 1, len(var1)))
    line = ax.add_collection(lc)
    
    # Mark start and end points
    ax.plot(var1[0], var2[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(var1[-1], var2[-1], 'ro', markersize=12, label='End', zorder=5)
    
    # Add direction arrows
    n_arrows = 5
    arrow_indices = np.linspace(0, len(var1)-2, n_arrows, dtype=int)
    for i in arrow_indices:
        dx = var1[i+1] - var1[i]
        dy = var2[i+1] - var2[i]
        ax.arrow(var1[i], var2[i], dx, dy, 
                head_width=0.02*abs(var1.max()-var1.min()),
                head_length=0.02*abs(var2.max()-var2.min()),
                fc='black', ec='black', alpha=0.5, zorder=4)
    
    ax.set_xlabel(var1_label, fontsize=13)
    ax.set_ylabel(var2_label, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for time
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Normalized Time', fontsize=11)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def plot_phase_space_3D(solution, title="3D Phase Space", filename=None):
    """
    Create 3D phase space plot (C_frozen, C_active, T_s)
    
    Parameters:
    -----------
    solution : array
        Solution array
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    set_plot_style()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    C_frozen = solution[:, 0]
    C_active = solution[:, 1]
    T_s = solution[:, 3]
    
    # Plot trajectory with color gradient
    time_colors = np.linspace(0, 1, len(C_frozen))
    scatter = ax.scatter(C_frozen, C_active, T_s, c=time_colors, 
                        cmap='viridis', s=20, alpha=0.6)
    
    # Plot trajectory line
    ax.plot(C_frozen, C_active, T_s, 'gray', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax.scatter(C_frozen[0], C_active[0], T_s[0], 
              c='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(C_frozen[-1], C_active[-1], T_s[-1], 
              c='red', s=200, marker='o', label='End', edgecolors='black')
    
    ax.set_xlabel('Frozen Carbon [PgC]', fontsize=12, labelpad=10)
    ax.set_ylabel('Active Carbon [PgC]', fontsize=12, labelpad=10)
    ax.set_zlabel('Temperature [°C]', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Normalized Time', fontsize=11)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


# ============================================================================
# ANALYSIS PLOTS
# ============================================================================

def plot_feedback_analysis(results_dict, title="Feedback Analysis", filename=None):
    """
    Compare final states across different feedback scenarios
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of scenario results
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    scenario_names = list(results_dict.keys())
    n_scenarios = len(scenario_names)
    
    # Extract final values
    final_C_frozen = []
    final_C_active = []
    final_C_atm = []
    final_T = []
    
    for scenario_name in scenario_names:
        t, solution = results_dict[scenario_name]
        final_C_frozen.append(solution[-1, 0])
        final_C_active.append(solution[-1, 1])
        final_C_atm.append(solution[-1, 2])
        final_T.append(solution[-1, 3])
    
    x_pos = np.arange(n_scenarios)
    
    # Bar plots for final values
    axes[0, 0].bar(x_pos, final_C_frozen, color='blue', alpha=0.7)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Final Frozen Carbon [PgC]')
    axes[0, 0].set_title('Frozen Permafrost Carbon (Final)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(x_pos, final_C_active, color='green', alpha=0.7)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Final Active Carbon [PgC]')
    axes[0, 1].set_title('Active Layer Carbon (Final)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].bar(x_pos, final_C_atm, color='red', alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Final Atmospheric CO₂ [PgC]')
    axes[1, 0].set_title('Atmospheric Carbon (Final)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=600, color='gray', linestyle='--', 
                      label='Pre-industrial', alpha=0.5)
    axes[1, 0].legend()
    
    axes[1, 1].bar(x_pos, final_T, color='orange', alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Final Temperature [°C]')
    axes[1, 1].set_title('Surface Temperature (Final)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', 
                      label='0°C reference', alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def plot_temperature_dependent_rates(T_range=None, params_dict=None, 
                                     title="Temperature-Dependent Rates", 
                                     filename=None):
    """
    Plot how rates vary with temperature
    
    Parameters:
    -----------
    T_range : array, optional
        Temperature range to plot
    params_dict : dict, optional
        Parameter dictionary
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    if T_range is None:
        T_range = np.linspace(-15, 15, 100)
    
    if params_dict is None:
        params_dict = params.get_default_params()
    
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Calculate rates
    k_thaw = np.array([model.thaw_rate(T, params_dict) for T in T_range])
    k_decomp = np.array([model.decomposition_rate(T, params_dict) for T in T_range])
    alpha_vals = np.array([model.albedo(T, params_dict) for T in T_range])
    
    # Assume constant C_atm for forcing calculation
    C_atm_test = 800  # Example value
    Delta_F = np.array([model.CO2_forcing(C_atm_test, params_dict) 
                       for _ in T_range])
    
    # Plot thaw rate
    axes[0, 0].plot(T_range, k_thaw, 'b-', linewidth=2.5)
    axes[0, 0].set_xlabel('Temperature [°C]')
    axes[0, 0].set_ylabel('Thaw Rate [1/year]')
    axes[0, 0].set_title('Permafrost Thawing Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot decomposition rate
    axes[0, 1].plot(T_range, k_decomp, 'g-', linewidth=2.5)
    axes[0, 1].set_xlabel('Temperature [°C]')
    axes[0, 1].set_ylabel('Decomposition Rate [1/year]')
    axes[0, 1].set_title('Carbon Decomposition Rate (Q10)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot albedo
    axes[1, 0].plot(T_range, alpha_vals, 'orange', linewidth=2.5)
    axes[1, 0].set_xlabel('Temperature [°C]')
    axes[1, 0].set_ylabel('Albedo')
    axes[1, 0].set_title('Ice-Albedo Feedback')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=params_dict['T_albedo_mid'], color='red', 
                      linestyle='--', label='Transition midpoint', alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # Plot combined effect (qualitative)
    axes[1, 1].plot(T_range, k_thaw * k_decomp, 'purple', linewidth=2.5)
    axes[1, 1].set_xlabel('Temperature [°C]')
    axes[1, 1].set_ylabel('Combined Rate Effect')
    axes[1, 1].set_title('Combined Thaw × Decomposition')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing visualization functions...")
    
    # Run a simple simulation
    params_dict = params.get_default_params()
    t, solution = model.run_simulation((0, 100), dt=0.5)
    
    # Test basic time series plot
    print("\n--- Creating time series plot ---")
    plot_time_series(t, solution, title="Test Time Series")
    
    print("\n✓ Visualization module complete!")
    print("Ready for creating plots!")