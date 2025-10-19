"""
extended_simulations.py
Arctic Permafrost-Carbon-Climate Feedback Model
Extended simulations (200-300 years) for long-term analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import PARAMS, SCENARIOS
from solver import run_model, save_results
from visualization import plot_time_series
import os


def run_extended_simulations(years=300, resolution=1001):
    """
    Run extended simulations out to 200-300 years
    
    Parameters:
        years : int
            Number of years to simulate
        resolution : int
            Number of time points
    
    Returns:
        results_dict : dict
            Results for all scenarios
    """
    
    print("=" * 70)
    print(f"EXTENDED SIMULATIONS: {years}-YEAR PROJECTIONS")
    print("=" * 70)
    
    # Setup
    initial_state = np.array([
        PARAMS['C_atm_0'],
        PARAMS['C_active_0'],
        PARAMS['C_deep_0'],
        PARAMS['T_s_0']
    ])
    
    time_points = np.linspace(0, years, resolution)
    
    print(f"\nSimulation period: 2000-{2000+years}")
    print(f"Time resolution: {time_points[1]-time_points[0]:.3f} years")
    
    # Run all scenarios
    scenarios = ['RCP2.6', 'RCP4.5', 'RCP8.5']
    results_dict = {}
    
    for scenario in scenarios:
        print(f"\nRunning {scenario}...")
        
        results = run_model(
            initial_state,
            time_points,
            PARAMS,
            forcing_scenario=scenario,
            use_albedo_feedback=False
        )
        
        results_dict[scenario] = results
        
        # Print summary
        print(f"  Year {2000+years}:")
        print(f"    CO2:         {results['CO2_ppm'][-1]:.1f} ppm")
        print(f"    Temperature: {results['T_s'][-1]:.2f} °C")
        print(f"    C_deep:      {results['C_deep'][-1]:.1f} Pg C")
        
        # Calculate changes
        delta_co2 = results['CO2_ppm'][-1] - results['CO2_ppm'][0]
        delta_T = results['T_s'][-1] - results['T_s'][0]
        
        print(f"  Total change:")
        print(f"    ΔCO2:        {delta_co2:+.1f} ppm")
        print(f"    ΔT:          {delta_T:+.2f} °C")
    
    return results_dict


def analyze_equilibration(results_dict):
    """
    Analyze how close system is to equilibrium by end of simulation
    
    Parameters:
        results_dict : dict
            Results from extended simulations
    """
    
    print("\n" + "=" * 70)
    print("EQUILIBRATION ANALYSIS")
    print("=" * 70)
    
    for scenario, results in results_dict.items():
        print(f"\n{scenario}:")
        
        # Calculate rate of change in final 50 years
        final_50yr = int(0.2 * len(results['time']))  # Last 20% of simulation
        
        co2_trend = np.polyfit(results['time'][-final_50yr:], 
                               results['CO2_ppm'][-final_50yr:], 1)[0]
        
        temp_trend = np.polyfit(results['time'][-final_50yr:], 
                                results['T_s'][-final_50yr:], 1)[0]
        
        print(f"  Final CO2 trend:  {co2_trend:+.3f} ppm/yr")
        print(f"  Final temp trend: {temp_trend:+.5f} °C/yr")
        
        # Check if approaching equilibrium
        if abs(co2_trend) < 0.1 and abs(temp_trend) < 0.001:
            print(f"  Status: ✓ Approaching equilibrium")
        else:
            print(f"  Status: ⟳ Still evolving")


def plot_extended_timeseries(results_dict, save_path=None):
    """
    Create publication-quality figure for extended simulations
    
    Parameters:
        results_dict : dict
            Results dictionary
        save_path : str, optional
            Path to save figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Extended Permafrost Model Projections (300 years)', 
                 fontsize=16, fontweight='bold')
    
    colors = {'RCP2.6': 'green', 'RCP4.5': 'orange', 'RCP8.5': 'red'}
    
    for scenario, results in results_dict.items():
        years = results['time'] + 2000
        color = colors[scenario]
        
        # CO2
        axes[0, 0].plot(years, results['CO2_ppm'], 
                       color=color, linewidth=2.5, label=scenario, alpha=0.8)
        
        # Temperature
        axes[0, 1].plot(years, results['T_s'], 
                       color=color, linewidth=2.5, label=scenario, alpha=0.8)
        
        # Total permafrost carbon
        C_perm = results['C_active'] + results['C_deep']
        axes[1, 0].plot(years, C_perm, 
                       color=color, linewidth=2.5, label=scenario, alpha=0.8)
        
        # Cumulative carbon loss
        C_perm_initial = PARAMS['C_active_0'] + PARAMS['C_deep_0']
        C_loss = C_perm_initial - C_perm
        axes[1, 1].plot(years, C_loss, 
                       color=color, linewidth=2.5, label=scenario, alpha=0.8)
    
    # Format axes
    axes[0, 0].set_ylabel('CO₂ Concentration (ppm)', fontsize=12)
    axes[0, 0].set_title('Atmospheric CO₂', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='upper left', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=280, color='gray', linestyle='--', alpha=0.5, label='Pre-industrial')
    
    axes[0, 1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0, 1].set_title('Arctic Surface Temperature', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='upper left', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Freezing')
    
    axes[1, 0].set_xlabel('Year', fontsize=12)
    axes[1, 0].set_ylabel('Carbon (Pg C)', fontsize=12)
    axes[1, 0].set_title('Total Permafrost Carbon', fontsize=13, fontweight='bold')
    axes[1, 0].legend(loc='upper right', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Year', fontsize=12)
    axes[1, 1].set_ylabel('Carbon Loss (Pg C)', fontsize=12)
    axes[1, 1].set_title('Cumulative Permafrost Carbon Release', fontsize=13, fontweight='bold')
    axes[1, 1].legend(loc='upper left', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig


def main():
    """Main execution"""
    
    # Create output directories
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)
    
    # Run 300-year simulations
    results_dict = run_extended_simulations(years=300, resolution=1501)
    
    # Analyze equilibration
    analyze_equilibration(results_dict)
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    for scenario, results in results_dict.items():
        filename = f'../results/{scenario.lower()}_extended_300yr.csv'
        save_results(results, filename)
    
    # Create figure
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)
    
    fig = plot_extended_timeseries(results_dict, 
                                    save_path='../figures/extended_300yr_projections.png')
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("EXTENDED SIMULATIONS COMPLETE")
    print("=" * 70)
    
    return results_dict


if __name__ == "__main__":
    results = main()