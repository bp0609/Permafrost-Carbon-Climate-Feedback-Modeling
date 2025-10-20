"""
phase5_comparative_visualizations.py
Tasks 5.7-5.8: Feedback Comparison and Sensitivity Heatmaps

Creates:
- Enhanced bar charts showing feedback contributions
- Sensitivity analysis heatmaps
- Parameter importance visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    from phase4_comparative_studies import run_feedback_scenarios
    from parameters import PARAMS
except:
    print("Phase 4 modules not found.")


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures"""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


# =============================================================================
# TASK 5.7: FEEDBACK COMPARISON BAR CHART
# =============================================================================

def create_feedback_contribution_chart(feedback_results, save_figure=True):
    """
    Create enhanced bar chart showing feedback contributions
    """
    
    setup_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = feedback_results['scenarios']
    
    # Extract final temperatures
    names = list(scenarios.keys())
    final_temps = [scenarios[name][-1, 3] for name in names]
    final_CO2 = [scenarios[name][-1, 0] / 2.124 for name in names]
    
    # Calculate contributions relative to baseline (no feedbacks)
    baseline_T = final_temps[0]  # No feedbacks
    baseline_CO2 = final_CO2[0]
    
    contributions_T = [T - baseline_T for T in final_temps]
    contributions_CO2 = [CO2 - baseline_CO2 for CO2 in final_CO2]
    
    # =========================================================================
    # Panel 1: Temperature Contributions
    # =========================================================================
    ax = axes[0]
    
    colors = ['#808080', '#2ecc71', '#f39c12', '#e74c3c']  # Gray, green, orange, red
    
    x = np.arange(len(names))
    bars = ax.bar(x, contributions_T, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, contrib) in enumerate(zip(bars, contributions_T)):
        height = bar.get_height()
        label = f'{contrib:+.2f}°C'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=10, fontweight='bold')
    
    # Add cumulative annotation
    carbon_contrib = contributions_T[1]  # Carbon only
    total_contrib = contributions_T[3]  # All feedbacks
    
    # Draw arrow showing total amplification
    ax.annotate('', xy=(3, total_contrib), xytext=(0, 0),
               arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(3.2, total_contrib/2, 
           f'Total\nAmplification:\n{total_contrib:.1f}°C',
           fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Temperature Contribution (°C)', fontsize=12, fontweight='bold')
    ax.set_title('(a) Feedback Contributions to Warming',
                fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel 2: CO₂ Contributions
    # =========================================================================
    ax = axes[1]
    
    bars = ax.bar(x, contributions_CO2, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, contrib) in enumerate(zip(bars, contributions_CO2)):
        height = bar.get_height()
        label = f'{contrib:+.0f} ppm'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=10, fontweight='bold')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('CO₂ Contribution (ppm)', fontsize=12, fontweight='bold')
    ax.set_title('(b) Feedback Contributions to CO₂',
                fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Quantifying Feedback Mechanism Contributions',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_feedback_contributions.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_feedback_contributions.png")
    
    return fig


def create_stacked_feedback_chart(feedback_results, save_figure=True):
    """
    Create stacked bar chart showing cumulative feedback effects
    """
    
    setup_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scenarios = feedback_results['scenarios']
    
    # Get final values
    names = list(scenarios.keys())
    final_temps = np.array([scenarios[name][-1, 3] for name in names])
    
    # Calculate incremental contributions
    baseline = final_temps[0]
    carbon_add = final_temps[1] - final_temps[0]
    albedo_add = final_temps[2] - final_temps[1] if len(final_temps) > 2 else 0
    synergy = final_temps[3] - final_temps[2] if len(final_temps) > 3 else 0
    
    # Create stacked bar
    categories = ['Baseline\n(no feedbacks)', 'Carbon\nfeedback', 
                 'Albedo\nfeedback', 'Synergy']
    values = [baseline, carbon_add, albedo_add, synergy]
    colors = ['#808080', '#2ecc71', '#f39c12', '#e74c3c']
    
    x = [0]
    bottom = 0
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        bar = ax.bar(x, val, bottom=bottom, color=color, alpha=0.8,
                    edgecolor='black', linewidth=1.5, label=cat)
        
        # Add value label
        if val != 0:
            ax.text(0, bottom + val/2, f'{val:+.1f}°C',
                   ha='center', va='center', fontsize=11,
                   fontweight='bold', color='white' if i > 0 else 'black')
        
        bottom += val
    
    # Add total line
    ax.axhline(bottom, color='red', linestyle='--', linewidth=2.5,
              label=f'Total = {bottom:.1f}°C')
    
    # Annotations
    ax.text(0.2, baseline/2, 'Direct\nforcing\nonly', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.text(0.2, baseline + carbon_add/2, '+Permafrost\ncarbon\nrelease',
           fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Feedback Effects: Building to Total Warming',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_stacked_feedbacks.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_stacked_feedbacks.png")
    
    return fig


# =============================================================================
# TASK 5.8: SENSITIVITY ANALYSIS HEATMAP
# =============================================================================

def perform_sensitivity_analysis(params_to_vary, save_figure=True):
    """
    Perform systematic sensitivity analysis and create heatmap
    
    Parameters:
        params_to_vary: dict of parameter names and their variation ranges
    """
    
    setup_publication_style()
    
    from model import permafrost_model
    from scipy.integrate import odeint
    
    # Define baseline forcing
    def baseline_forcing(t):
        return 7.0 + 0.25*t if t < 40 else 17.0
    
    # Baseline parameters
    base_params = PARAMS.copy()
    initial_state = np.array([594.0, 174.0, 800.0, -2.0])
    t = np.linspace(0, 100, 500)
    
    # Define RHS
    def rhs_baseline(state, time):
        return permafrost_model(state, time, base_params, baseline_forcing, True)
    
    # Run baseline
    sol_baseline = odeint(rhs_baseline, initial_state, t)
    T_baseline_final = sol_baseline[-1, 3]
    CO2_baseline_final = sol_baseline[-1, 0]
    
    # Storage for sensitivity results
    param_names = list(params_to_vary.keys())
    n_params = len(param_names)
    n_variations = 5
    
    sensitivity_matrix_T = np.zeros((n_params, n_variations))
    sensitivity_matrix_CO2 = np.zeros((n_params, n_variations))
    variation_factors = np.linspace(0.5, 1.5, n_variations)
    
    print("Performing sensitivity analysis...")
    
    for i, param_name in enumerate(param_names):
        print(f"  Varying {param_name}...")
        
        for j, factor in enumerate(variation_factors):
            # Create modified parameters
            params_modified = base_params.copy()
            params_modified[param_name] = base_params[param_name] * factor
            
            # Define RHS with modified parameters
            def rhs_modified(state, time):
                return permafrost_model(state, time, params_modified,
                                      baseline_forcing, True)
            
            try:
                # Run simulation
                sol = odeint(rhs_modified, initial_state, t)
                T_final = sol[-1, 3]
                CO2_final = sol[-1, 0]
                
                # Store relative change
                sensitivity_matrix_T[i, j] = (T_final - T_baseline_final) / T_baseline_final * 100
                sensitivity_matrix_CO2[i, j] = (CO2_final - CO2_baseline_final) / CO2_baseline_final * 100
            except:
                sensitivity_matrix_T[i, j] = np.nan
                sensitivity_matrix_CO2[i, j] = np.nan
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Temperature sensitivity heatmap
    ax = axes[0]
    im1 = ax.imshow(sensitivity_matrix_T, cmap='RdBu_r', aspect='auto',
                   vmin=-20, vmax=20)
    
    ax.set_xticks(np.arange(n_variations))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels([f'{f:.1f}×' for f in variation_factors])
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Parameter Multiplier', fontweight='bold')
    ax.set_title('(a) Temperature Sensitivity (%)', fontweight='bold', loc='left')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('Relative change in final T (%)', fontweight='bold')
    
    # Add value annotations
    for i in range(n_params):
        for j in range(n_variations):
            val = sensitivity_matrix_T[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}',
                             ha="center", va="center",
                             color="white" if abs(val) > 10 else "black",
                             fontsize=8)
    
    # CO2 sensitivity heatmap
    ax = axes[1]
    im2 = ax.imshow(sensitivity_matrix_CO2, cmap='RdBu_r', aspect='auto',
                   vmin=-20, vmax=20)
    
    ax.set_xticks(np.arange(n_variations))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels([f'{f:.1f}×' for f in variation_factors])
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Parameter Multiplier', fontweight='bold')
    ax.set_title('(b) CO₂ Sensitivity (%)', fontweight='bold', loc='left')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('Relative change in final CO₂ (%)', fontweight='bold')
    
    # Add value annotations
    for i in range(n_params):
        for j in range(n_variations):
            val = sensitivity_matrix_CO2[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}',
                             ha="center", va="center",
                             color="white" if abs(val) > 10 else "black",
                             fontsize=8)
    
    fig.suptitle('Parameter Sensitivity Analysis: Impact on Final State',
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_sensitivity_heatmap.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_sensitivity_heatmap.png")
    
    return fig, sensitivity_matrix_T, sensitivity_matrix_CO2


def create_tornado_diagram(params_to_vary, save_figure=True):
    """
    Create tornado diagram showing parameter importance
    """
    
    setup_publication_style()
    
    from model import permafrost_model
    from scipy.integrate import odeint
    
    # Run baseline
    def baseline_forcing(t):
        return 7.0 + 0.25*t if t < 40 else 17.0
    
    base_params = PARAMS.copy()
    initial_state = np.array([594.0, 174.0, 800.0, -2.0])
    t = np.linspace(0, 100, 500)
    
    def rhs_baseline(state, time):
        return permafrost_model(state, time, base_params, baseline_forcing, True)
    
    sol_baseline = odeint(rhs_baseline, initial_state, t)
    T_baseline = sol_baseline[-1, 3]
    
    # Calculate sensitivities
    param_names = list(params_to_vary.keys())
    low_values = []
    high_values = []
    
    print("Calculating parameter importance...")
    
    for param_name in param_names:
        print(f"  Testing {param_name}...")
        
        # Low value (0.8x baseline)
        params_low = base_params.copy()
        params_low[param_name] = base_params[param_name] * 0.8
        
        def rhs_low(state, time):
            return permafrost_model(state, time, params_low, baseline_forcing, True)
        
        sol_low = odeint(rhs_low, initial_state, t)
        T_low = sol_low[-1, 3]
        
        # High value (1.2x baseline)
        params_high = base_params.copy()
        params_high[param_name] = base_params[param_name] * 1.2
        
        def rhs_high(state, time):
            return permafrost_model(state, time, params_high, baseline_forcing, True)
        
        sol_high = odeint(rhs_high, initial_state, t)
        T_high = sol_high[-1, 3]
        
        low_values.append(T_low - T_baseline)
        high_values.append(T_high - T_baseline)
    
    # Create tornado diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(param_names))
    
    # Plot bars
    for i, (low, high) in enumerate(zip(low_values, high_values)):
        ax.barh(i, low, left=0, height=0.8, color='blue', alpha=0.6)
        ax.barh(i, high, left=0, height=0.8, color='red', alpha=0.6)
    
    # Sort by total range
    total_range = [abs(h - l) for h, l in zip(high_values, low_values)]
    sorted_indices = np.argsort(total_range)[::-1]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([param_names[i] for i in sorted_indices])
    ax.set_xlabel('Temperature Change from Baseline (°C)', fontweight='bold')
    ax.set_title('Tornado Diagram: Parameter Importance\n(Blue: -20%, Red: +20%)',
                fontweight='bold', fontsize=14)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.6, label='-20% from baseline'),
                      Patch(facecolor='red', alpha=0.6, label='+20% from baseline')]
    ax.legend(handles=legend_elements, loc='best')
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_tornado_diagram.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_tornado_diagram.png")
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Generate all comparative visualizations
    """
    
    print("\n" + "="*70)
    print("PHASE 5: COMPARATIVE VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Task 5.7: Feedback comparison
    print("Running feedback scenarios...")
    feedback_results = run_feedback_scenarios(
        forcing_scenario='RCP4.5',
        time_span=(0, 100)
    )
    
    print("\nCreating feedback visualizations...\n")
    
    print("1. Feedback contribution chart...")
    fig1 = create_feedback_contribution_chart(feedback_results, save_figure=True)
    
    print("\n2. Stacked feedback chart...")
    fig2 = create_stacked_feedback_chart(feedback_results, save_figure=True)
    
    # Task 5.8: Sensitivity analysis
    print("\n3. Sensitivity heatmap...")
    params_to_vary = {
        'k_0': PARAMS['k_0'],
        'Q_10': PARAMS['Q_10'],
        'E_a': PARAMS['E_a'],
        'a_co2': PARAMS['a_co2'],
        'k_uptake': PARAMS['k_uptake'],
    }
    
    fig3, sens_T, sens_CO2 = perform_sensitivity_analysis(params_to_vary,
                                                          save_figure=True)
    
    print("\n4. Tornado diagram...")
    fig4 = create_tornado_diagram(params_to_vary, save_figure=True)
    
    print("\n" + "="*70)
    print("COMPARATIVE VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated 4 publication-quality figures:")
    print("  - enhanced_feedback_contributions.png")
    print("  - enhanced_stacked_feedbacks.png")
    print("  - enhanced_sensitivity_heatmap.png")
    print("  - enhanced_tornado_diagram.png")
    print("\n" + "="*70 + "\n")
    
    plt.show()
    
    return feedback_results, sens_T, sens_CO2


if __name__ == "__main__":
    results = main()