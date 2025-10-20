"""
phase5_enhanced_timeseries.py
Tasks 5.1-5.2: Enhanced Multi-Panel Time Series Plots

Creates publication-ready time series visualizations with:
- Professional formatting
- Consistent styling
- Informative annotations
- Multiple layout options
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import sys
import os

sys.path.append(os.path.dirname(__file__))

# Try to load existing Phase 4 results
try:
    from baseline_simulations import run_baseline_simulations, get_forcing_scenarios
except:
    print("Phase 4 baseline module not found. Will generate data fresh.")


# =============================================================================
# PUBLICATION-QUALITY STYLE SETTINGS
# =============================================================================

def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures
    """
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.0,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def get_scenario_colors():
    """
    Define consistent color scheme for scenarios
    """
    return {
        'RCP2.6': '#2166ac',  # Blue - coolest
        'RCP4.5': '#fdae61',  # Orange - moderate
        'RCP8.5': '#d73027',  # Red - warmest
        'Pulse': '#4d4d4d'    # Gray - special case
    }


def get_scenario_linestyles():
    """
    Define line styles for scenarios (useful for B&W printing)
    """
    return {
        'RCP2.6': '-',
        'RCP4.5': '--',
        'RCP8.5': '-.',
        'Pulse': ':'
    }


# =============================================================================
# TASK 5.1: ENHANCED MULTI-PANEL TIME SERIES
# =============================================================================

def create_enhanced_timeseries(results_dict, save_figure=True):
    """
    Create publication-ready multi-panel time series
    
    Enhanced features:
    - Professional styling
    - Shaded uncertainty regions (if available)
    - Key event markers
    - Annotated critical points
    - Consistent color scheme
    """
    
    setup_publication_style()
    
    # Create figure with custom gridspec for better control
    fig = plt.figure(figsize=(12, 11))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
    
    colors = get_scenario_colors()
    linestyles = get_scenario_linestyles()
    
    # =========================================================================
    # Panel 1: Atmospheric CO₂
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    
    for scenario_name, results in results_dict.items():
        CO2_ppm = results['C_atm'] / 2.124
        ax1.plot(results['t'], CO2_ppm, 
                color=colors[scenario_name],
                linestyle=linestyles[scenario_name],
                linewidth=2.5,
                label=scenario_name,
                alpha=0.9)
    
    # Add reference lines
    ax1.axhline(y=280, color='green', linestyle=':', alpha=0.6, linewidth=1.5,
               label='Pre-industrial (280 ppm)')
    ax1.axhline(y=420, color='orange', linestyle=':', alpha=0.6, linewidth=1.5,
               label='~Current (420 ppm)')
    
    # Shade policy-relevant zones
    ax1.axhspan(280, 350, alpha=0.05, color='green', zorder=0)
    ax1.axhspan(450, 600, alpha=0.05, color='red', zorder=0)
    
    ax1.set_ylabel('Atmospheric CO₂ (ppm)', fontweight='bold')
    ax1.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)
    
    # Add subplot label
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel 2: Surface Temperature
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    
    for scenario_name, results in results_dict.items():
        ax2.plot(results['t'], results['T_s'],
                color=colors[scenario_name],
                linestyle=linestyles[scenario_name],
                linewidth=2.5,
                label=scenario_name,
                alpha=0.9)
    
    # Add critical thresholds
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
               label='Freezing point')
    ax2.axhline(y=2, color='red', linestyle=':', alpha=0.5, linewidth=1.5,
               label='Paris Agreement target')
    
    # Shade temperature zones
    ax2.axhspan(-5, 0, alpha=0.05, color='blue', zorder=0)
    ax2.axhspan(2, 10, alpha=0.05, color='red', zorder=0)
    
    ax2.set_ylabel('Surface Temperature (°C)', fontweight='bold')
    ax2.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 200)
    
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes,
            fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel 3: Active Layer Carbon
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    
    for scenario_name, results in results_dict.items():
        ax3.plot(results['t'], results['C_active'],
                color=colors[scenario_name],
                linestyle=linestyles[scenario_name],
                linewidth=2.5,
                label=scenario_name,
                alpha=0.9)
    
    # Mark initial value
    initial_value = list(results_dict.values())[0]['C_active'][0]
    ax3.axhline(y=initial_value, color='gray', linestyle=':', 
               alpha=0.5, linewidth=1.5, label=f'Initial ({initial_value:.0f} Pg C)')
    
    ax3.set_ylabel('Active Layer Carbon (Pg C)', fontweight='bold')
    ax3.legend(loc='best', ncol=2, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200)
    
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes,
            fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel 4: Deep Permafrost Carbon
    # =========================================================================
    ax4 = fig.add_subplot(gs[3])
    
    for scenario_name, results in results_dict.items():
        ax4.plot(results['t'], results['C_deep'],
                color=colors[scenario_name],
                linestyle=linestyles[scenario_name],
                linewidth=2.5,
                label=scenario_name,
                alpha=0.9)
    
    # Mark initial value
    initial_value = list(results_dict.values())[0]['C_deep'][0]
    ax4.axhline(y=initial_value, color='gray', linestyle=':', 
               alpha=0.5, linewidth=1.5, label=f'Initial ({initial_value:.0f} Pg C)')
    
    ax4.set_ylabel('Deep Permafrost Carbon (Pg C)', fontweight='bold')
    ax4.set_xlabel('Time (years from 2000)', fontweight='bold', fontsize=12)
    ax4.legend(loc='best', ncol=2, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 200)
    
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes,
            fontsize=14, fontweight='bold', va='top')
    
    # Overall title
    fig.suptitle('Arctic Permafrost-Carbon-Climate System Evolution\nUnder Different Emission Scenarios',
                fontsize=15, fontweight='bold', y=0.995)
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_timeseries_4panel.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_timeseries_4panel.png")
    
    return fig


def create_side_by_side_comparison(results_dict, save_figure=True):
    """
    Alternative layout: Side-by-side comparison of scenarios
    """
    
    setup_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = get_scenario_colors()
    
    scenarios = list(results_dict.keys())
    
    # Plot each scenario in its own subplot
    for idx, scenario_name in enumerate(scenarios):
        if idx >= 4:
            break
        
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        results = results_dict[scenario_name]
        
        # Create twin axes
        ax2 = ax.twinx()
        
        # Temperature on left axis
        line1 = ax.plot(results['t'], results['T_s'], 
                       color='red', linewidth=2.5, label='Temperature')
        ax.set_ylabel('Temperature (°C)', fontweight='bold', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        
        # CO2 on right axis
        line2 = ax2.plot(results['t'], results['C_atm']/2.124,
                        color='blue', linewidth=2.5, label='CO₂')
        ax2.set_ylabel('CO₂ (ppm)', fontweight='bold', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax.set_xlabel('Time (years)', fontweight='bold')
        ax.set_title(scenario_name, fontweight='bold', fontsize=13,
                    color=colors[scenario_name])
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)
        
        # Add subplot label
        labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(0.02, 0.98, labels[idx], transform=ax.transAxes,
               fontsize=13, fontweight='bold', va='top')
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_timeseries_comparison.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_timeseries_comparison.png")
    
    return fig


def create_animated_style_plot(results_dict, save_figure=True):
    """
    Create a plot showing temporal progression with gradient coloring
    """
    
    setup_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Temperature vs CO2 with time gradient
    ax = axes[0]
    
    for scenario_name, results in results_dict.items():
        t = results['t']
        T = results['T_s']
        CO2 = results['C_atm'] / 2.124
        
        # Create scatter plot with time-based coloring
        scatter = ax.scatter(CO2, T, c=t, cmap='plasma', s=10, alpha=0.6)
        
        # Add arrows to show direction
        n_arrows = 5
        indices = np.linspace(10, len(t)-10, n_arrows, dtype=int)
        for i in indices:
            dx = CO2[i+5] - CO2[i]
            dy = T[i+5] - T[i]
            ax.arrow(CO2[i], T[i], dx, dy,
                    head_width=5, head_length=0.2,
                    fc='black', ec='black', alpha=0.3)
        
        # Label start and end
        ax.plot(CO2[0], T[0], 'go', markersize=10, 
               markeredgewidth=2, markeredgecolor='darkgreen',
               label=f'{scenario_name} (start)')
        ax.plot(CO2[-1], T[-1], 'rs', markersize=10,
               markeredgewidth=2, markeredgecolor='darkred',
               label=f'{scenario_name} (end)')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (years)', fontweight='bold')
    
    ax.set_xlabel('Atmospheric CO₂ (ppm)', fontweight='bold')
    ax.set_ylabel('Surface Temperature (°C)', fontweight='bold')
    ax.set_title('System Trajectory: Temperature vs. CO₂\n(Color shows time progression)',
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # Panel 2: Permafrost loss rate
    ax = axes[1]
    
    colors = get_scenario_colors()
    
    for scenario_name, results in results_dict.items():
        t = results['t']
        total_permafrost = results['C_active'] + results['C_deep']
        
        # Calculate loss rate
        loss_rate = -np.gradient(total_permafrost, t)
        
        ax.plot(t, loss_rate, color=colors[scenario_name],
               linewidth=2.5, label=scenario_name)
    
    ax.set_xlabel('Time (years)', fontweight='bold')
    ax.set_ylabel('Permafrost Carbon Loss Rate (Pg C/yr)', fontweight='bold')
    ax.set_title('Permafrost Degradation Rate Over Time',
                fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim(0, 200)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('../figures', exist_ok=True)
        plt.savefig('../figures/enhanced_timeseries_animated.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: ../figures/enhanced_timeseries_animated.png")
    
    return fig


# =============================================================================
# TASK 5.2: FORMATTING AND STYLING
# =============================================================================

def create_formatted_legend_demo():
    """
    Demonstrate various legend and annotation options
    """
    
    setup_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate sample data
    t = np.linspace(0, 200, 200)
    
    # Style 1: Classic with box
    ax = axes[0, 0]
    for i, (name, color) in enumerate(get_scenario_colors().items()):
        ax.plot(t, np.sin(t/20 + i), color=color, linewidth=2, label=name)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    ax.set_title('Style 1: Boxed Legend', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Style 2: Transparent background
    ax = axes[0, 1]
    for i, (name, color) in enumerate(get_scenario_colors().items()):
        ax.plot(t, np.sin(t/20 + i), color=color, linewidth=2, label=name)
    ax.legend(loc='upper right', framealpha=0.5, fancybox=True)
    ax.set_title('Style 2: Transparent Legend', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Style 3: Outside plot area
    ax = axes[1, 0]
    for i, (name, color) in enumerate(get_scenario_colors().items()):
        ax.plot(t, np.sin(t/20 + i), color=color, linewidth=2, label=name)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Style 3: External Legend', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Style 4: Annotated with arrows
    ax = axes[1, 1]
    for i, (name, color) in enumerate(get_scenario_colors().items()):
        line, = ax.plot(t, np.sin(t/20 + i), color=color, linewidth=2)
        # Add annotation
        idx = 150 + i*10
        ax.annotate(name, xy=(t[idx], np.sin(t[idx]/20 + i)),
                   xytext=(20, 10*i), textcoords='offset points',
                   ha='left', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 color=color))
    ax.set_title('Style 4: Annotated Lines', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Generate all enhanced time series visualizations
    """
    
    print("\n" + "="*70)
    print("PHASE 5: ENHANCED TIME SERIES VISUALIZATION")
    print("="*70 + "\n")
    
    # Load or generate data
    print("Loading Phase 4 baseline results...")
    try:
        results_dict = run_baseline_simulations(
            time_span=(0, 200),
            dt=0.1,
            save_results=False
        )
        print("✓ Data loaded successfully\n")
    except Exception as e:
        print(f"Could not load Phase 4 data: {e}")
        print("Please run Phase 4 baseline simulations first.")
        return None
    
    # Create visualizations
    print("Creating enhanced visualizations...\n")
    
    print("1. Multi-panel time series (4 panels)...")
    fig1 = create_enhanced_timeseries(results_dict, save_figure=True)
    
    print("\n2. Side-by-side scenario comparison...")
    fig2 = create_side_by_side_comparison(results_dict, save_figure=True)
    
    print("\n3. Animated-style trajectory plot...")
    fig3 = create_animated_style_plot(results_dict, save_figure=True)
    
    print("\n" + "="*70)
    print("ENHANCED TIME SERIES COMPLETE")
    print("="*70)
    print("\nGenerated 3 publication-quality figures:")
    print("  - enhanced_timeseries_4panel.png")
    print("  - enhanced_timeseries_comparison.png")
    print("  - enhanced_timeseries_animated.png")
    print("\n" + "="*70 + "\n")
    
    plt.show()
    
    return results_dict


if __name__ == "__main__":
    results = main()