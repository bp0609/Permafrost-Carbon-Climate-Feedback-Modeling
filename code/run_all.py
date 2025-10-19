"""
run_all.py
Arctic Permafrost-Carbon-Climate Feedback Model
Main execution script - runs all scenarios and creates all figures
"""

import numpy as np
from parameters import PARAMS
from solver import run_model, run_multiple_scenarios, save_results
from visualization import (plot_time_series, plot_phase_portrait, 
                           plot_carbon_budget, plot_feedback_comparison)
import os


def main():
    """
    Main execution function - runs complete analysis pipeline
    """
    
    print("=" * 70)
    print("ARCTIC PERMAFROST-CARBON-CLIMATE FEEDBACK MODEL")
    print("Complete Analysis Pipeline")
    print("=" * 70)
    
    # Create output directories if they don't exist
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    print("\n1. SETUP")
    print("-" * 70)
    
    # Initial conditions
    initial_state = np.array([
        PARAMS['C_atm_0'],
        PARAMS['C_active_0'],
        PARAMS['C_deep_0'],
        PARAMS['T_s_0']
    ])
    
    print(f"Initial state (Year 2000):")
    print(f"  C_atm:    {initial_state[0]:.1f} Pg C")
    print(f"  C_active: {initial_state[1]:.1f} Pg C")
    print(f"  C_deep:   {initial_state[2]:.1f} Pg C")
    print(f"  T_s:      {initial_state[3]:.1f} °C")
    
    # Time span: 2000-2100
    time_points = np.linspace(0, 100, 1001)
    print(f"\nSimulation: 2000-2100 ({len(time_points)} time points)")
    
    # =========================================================================
    # RUN ALL SCENARIOS (No Albedo Feedback)
    # =========================================================================
    
    print("\n2. RUNNING ALL SCENARIOS (Baseline)")
    print("-" * 70)
    
    scenarios = ['RCP2.6', 'RCP4.5', 'RCP8.5']
    results_baseline = run_multiple_scenarios(
        initial_state,
        time_points,
        PARAMS,
        scenarios=scenarios,
        use_albedo_feedback=False
    )
    
    # Print summary
    print("\nResults Summary (Year 2100):")
    print(f"{'Scenario':<10} {'CO2 (ppm)':<12} {'ΔCO2':<10} {'Temp (°C)':<12} {'ΔT':<10}")
    print("-" * 60)
    
    for scenario in scenarios:
        res = results_baseline[scenario]
        co2_final = res['CO2_ppm'][-1]
        co2_change = co2_final - res['CO2_ppm'][0]
        t_final = res['T_s'][-1]
        t_change = t_final - res['T_s'][0]
        
        print(f"{scenario:<10} {co2_final:<12.1f} {co2_change:<10.1f} "
              f"{t_final:<12.2f} {t_change:<10.2f}")
    
    # =========================================================================
    # RUN WITH ALBEDO FEEDBACK (RCP 8.5 only for comparison)
    # =========================================================================
    
    print("\n3. RUNNING WITH ALBEDO FEEDBACK (RCP 8.5)")
    print("-" * 70)
    
    results_feedback = run_model(
        initial_state,
        time_points,
        PARAMS,
        forcing_scenario='RCP8.5',
        use_albedo_feedback=True
    )
    
    print(f"RCP 8.5 with albedo feedback:")
    print(f"  Final CO2:  {results_feedback['CO2_ppm'][-1]:.1f} ppm")
    print(f"  Final Temp: {results_feedback['T_s'][-1]:.2f} °C")
    
    # Compare with baseline
    baseline_8_5 = results_baseline['RCP8.5']
    co2_diff = results_feedback['CO2_ppm'][-1] - baseline_8_5['CO2_ppm'][-1]
    t_diff = results_feedback['T_s'][-1] - baseline_8_5['T_s'][-1]
    
    print(f"\nAlbedo feedback amplification:")
    print(f"  Additional CO2:  {co2_diff:+.1f} ppm")
    print(f"  Additional temp: {t_diff:+.2f} °C")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    print("\n4. SAVING RESULTS")
    print("-" * 70)
    
    for scenario in scenarios:
        filename = f'../results/{scenario.lower()}_baseline.csv'
        save_results(results_baseline[scenario], filename)
    
    save_results(results_feedback, '../results/rcp85_with_feedback.csv')
    
    # =========================================================================
    # CREATE VISUALIZATIONS
    # =========================================================================
    
    print("\n5. CREATING FIGURES")
    print("-" * 70)
    
    # Figure 1: Time series comparison
    print("  Creating Figure 1: Time series comparison...")
    plot_time_series(results_baseline, save_path='../figures/fig1_timeseries.png')
    
    # Figure 2: Phase portrait (RCP 4.5)
    print("  Creating Figure 2: Phase portrait...")
    plot_phase_portrait(results_baseline['RCP4.5'], 
                       state_vars=('C_atm', 'T_s'),
                       save_path='../figures/fig2_phase_portrait.png')
    
    # Figure 3: Carbon budget (RCP 8.5)
    print("  Creating Figure 3: Carbon budget...")
    plot_carbon_budget(results_baseline['RCP8.5'],
                      save_path='../figures/fig3_carbon_budget.png')
    
    # Figure 4: Feedback comparison
    print("  Creating Figure 4: Albedo feedback impact...")
    plot_feedback_comparison(baseline_8_5, results_feedback,
                            save_path='../figures/fig4_feedback_comparison.png')
    
    # =========================================================================
    # CALCULATE KEY METRICS
    # =========================================================================
    
    print("\n6. KEY METRICS")
    print("-" * 70)
    
    print("\nPermafrost carbon loss by 2100:")
    for scenario in scenarios:
        res = results_baseline[scenario]
        C_perm_initial = PARAMS['C_active_0'] + PARAMS['C_deep_0']
        C_perm_final = res['C_active'][-1] + res['C_deep'][-1]
        C_loss = C_perm_initial - C_perm_final
        pct_loss = 100 * C_loss / C_perm_initial
        
        print(f"  {scenario}: {C_loss:.1f} Pg C ({pct_loss:.1f}% of initial)")
    
    print("\nPermafrost feedback strength:")
    for scenario in scenarios:
        res = results_baseline[scenario]
        C_perm_loss = (PARAMS['C_active_0'] + PARAMS['C_deep_0']) - \
                      (res['C_active'][-1] + res['C_deep'][-1])
        T_increase = res['T_s'][-1] - res['T_s'][0]
        
        if T_increase > 0:
            sensitivity = C_perm_loss / T_increase
            print(f"  {scenario}: {sensitivity:.1f} Pg C per °C warming")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutput files created:")
    print("  Results: ../results/*.csv")
    print("  Figures: ../figures/*.png")
    print("\nNext steps:")
    print("  1. Review figures in ../figures/")
    print("  2. Analyze results in ../results/")
    print("  3. Run sensitivity analysis (see test_sensitivity.py)")
    print("  4. Perform stability analysis (see analysis.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()