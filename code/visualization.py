"""
Visualization functions for creating all plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_time_series(results, save_path):
    """
    Create 4-panel time series plot
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Colors for scenarios
    colors = {
        'RCP 2.6 (Low Emissions)': '#2E7D32',
        'Baseline': '#1976D2',
        'RCP 8.5 (High Emissions)': '#C62828'
    }
    
    # Panel 1: Atmospheric Carbon
    ax1 = fig.add_subplot(gs[0, 0])
    for scenario, data in results.items():
        t = data['t']
        C_atm = data['solution'][:, 0]
        ax1.plot(t, C_atm, label=scenario, color=colors[scenario], linewidth=2)
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('Atmospheric Carbon (Pg C)', fontsize=12)
    ax1.set_title('(a) Atmospheric Carbon', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Panel 2: Surface Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    for scenario, data in results.items():
        t = data['t']
        T = data['solution'][:, 3] - 273  # Convert to Celsius
        ax2.plot(t, T, label=scenario, color=colors[scenario], linewidth=2)
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title('(b) Surface Temperature', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Freezing point')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Panel 3: Active Layer Carbon
    ax3 = fig.add_subplot(gs[1, 0])
    for scenario, data in results.items():
        t = data['t']
        C_active = data['solution'][:, 1]
        ax3.plot(t, C_active, label=scenario, color=colors[scenario], linewidth=2)
    ax3.set_xlabel('Time (years)', fontsize=12)
    ax3.set_ylabel('Active Layer Carbon (Pg C)', fontsize=12)
    ax3.set_title('(c) Active Layer Permafrost', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Panel 4: Deep Permafrost Carbon
    ax4 = fig.add_subplot(gs[1, 1])
    for scenario, data in results.items():
        t = data['t']
        C_deep = data['solution'][:, 2]
        ax4.plot(t, C_deep, label=scenario, color=colors[scenario], linewidth=2)
    ax4.set_xlabel('Time (years)', fontsize=12)
    ax4.set_ylabel('Deep Permafrost Carbon (Pg C)', fontsize=12)
    ax4.set_title('(d) Deep Permafrost', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.suptitle('Permafrost-Carbon-Climate Model: Time Series Results',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved: {save_path}")
    plt.close()


def plot_phase_portrait(results, save_path):
    """
    Create phase portrait (C_atm vs T)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'RCP 2.6 (Low Emissions)': '#2E7D32',
        'Baseline': '#1976D2',
        'RCP 8.5 (High Emissions)': '#C62828'
    }
    
    for scenario, data in results.items():
        solution = data['solution']
        C_atm = solution[:, 0]
        T = solution[:, 3] - 273  # Celsius
        
        # Plot trajectory
        ax.plot(C_atm, T, label=scenario, color=colors[scenario], linewidth=2, alpha=0.7)
        
        # Mark start and end
        ax.plot(C_atm[0], T[0], 'o', color=colors[scenario], markersize=8, label=f'{scenario} (start)')
        ax.plot(C_atm[-1], T[-1], 's', color=colors[scenario], markersize=8, label=f'{scenario} (end)')
    
    ax.set_xlabel('Atmospheric Carbon (Pg C)', fontsize=13)
    ax.set_ylabel('Temperature (°C)', fontsize=13)
    ax.set_title('Phase Portrait: Temperature vs Atmospheric Carbon', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.legend(fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase portrait saved: {save_path}")
    plt.close()


def plot_bifurcation(bifurcation_data, save_path):
    """
    Create bifurcation diagram
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    emissions = bifurcation_data['emissions']
    T_eq = bifurcation_data['T_equilibria'] - 273  # Celsius
    C_atm_eq = bifurcation_data['C_atm_equilibria']
    stabilities = bifurcation_data['stabilities']
    
    # Separate stable and unstable
    stable_mask = np.array(stabilities) == "Stable"
    unstable_mask = np.array(stabilities) == "Unstable"
    saddle_mask = np.array(stabilities) == "Saddle"
    
    # Plot 1: Temperature bifurcation
    if np.any(stable_mask):
        ax1.plot(emissions[stable_mask], T_eq[stable_mask], 'o-', 
                color='blue', markersize=6, linewidth=2, label='Stable')
    if np.any(unstable_mask):
        ax1.plot(emissions[unstable_mask], T_eq[unstable_mask], 'o--',
                color='red', markersize=6, linewidth=2, label='Unstable')
    if np.any(saddle_mask):
        ax1.plot(emissions[saddle_mask], T_eq[saddle_mask], 'o:',
                color='orange', markersize=6, linewidth=2, label='Saddle')
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1, label='Freezing point')
    ax1.set_xlabel('Anthropogenic Emissions (Pg C/year)', fontsize=13)
    ax1.set_ylabel('Equilibrium Temperature (°C)', fontsize=13)
    ax1.set_title('(a) Temperature Bifurcation Diagram', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Atmospheric carbon bifurcation
    if np.any(stable_mask):
        ax2.plot(emissions[stable_mask], C_atm_eq[stable_mask], 'o-',
                color='blue', markersize=6, linewidth=2, label='Stable')
    if np.any(unstable_mask):
        ax2.plot(emissions[unstable_mask], C_atm_eq[unstable_mask], 'o--',
                color='red', markersize=6, linewidth=2, label='Unstable')
    if np.any(saddle_mask):
        ax2.plot(emissions[saddle_mask], C_atm_eq[saddle_mask], 'o:',
                color='orange', markersize=6, linewidth=2, label='Saddle')
    
    ax2.set_xlabel('Anthropogenic Emissions (Pg C/year)', fontsize=13)
    ax2.set_ylabel('Equilibrium Atmospheric Carbon (Pg C)', fontsize=13)
    ax2.set_title('(b) Atmospheric Carbon Bifurcation Diagram', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.suptitle('Bifurcation Analysis: System Response to Emissions',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Bifurcation diagram saved: {save_path}")
    plt.close()


def plot_feedback_comparison(feedback_analysis, save_path):
    """
    Create bar chart comparing feedback strengths
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = list(feedback_analysis.keys())
    colors = ['#2E7D32', '#1976D2', '#C62828']
    
    # Temperature change
    ax1 = axes[0, 0]
    delta_T = [feedback_analysis[s]['delta_T'] for s in scenarios]
    ax1.bar(range(len(scenarios)), delta_T, color=colors)
    ax1.set_ylabel('Temperature Change (K)', fontsize=12)
    ax1.set_title('(a) Temperature Change', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Carbon released
    ax2 = axes[0, 1]
    carbon_released = [feedback_analysis[s]['carbon_released'] for s in scenarios]
    ax2.bar(range(len(scenarios)), carbon_released, color=colors)
    ax2.set_ylabel('Carbon Released (Pg C)', fontsize=12)
    ax2.set_title('(b) Permafrost Carbon Released', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Permafrost loss fraction
    ax3 = axes[1, 0]
    loss_fraction = [feedback_analysis[s]['permafrost_loss_fraction'] * 100 for s in scenarios]
    ax3.bar(range(len(scenarios)), loss_fraction, color=colors)
    ax3.set_ylabel('Permafrost Loss (%)', fontsize=12)
    ax3.set_title('(c) Deep Permafrost Loss', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Atmospheric carbon change
    ax4 = axes[1, 1]
    delta_C = [feedback_analysis[s]['delta_C_atm'] for s in scenarios]
    ax4.bar(range(len(scenarios)), delta_C, color=colors)
    ax4.set_ylabel('Atmospheric C Change (Pg C)', fontsize=12)
    ax4.set_title('(d) Atmospheric Carbon Change', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feedback Analysis: Scenario Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feedback comparison saved: {save_path}")
    plt.close()