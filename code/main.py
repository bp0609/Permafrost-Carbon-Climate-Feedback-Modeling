"""
Main script to run all analyses
"""

import sys
import os
import numpy as np

# Import modules
from parameters import get_params_dict, get_initial_conditions
from solver import run_all_scenarios, save_results
from analysis import bifurcation_analysis, analyze_feedbacks, save_analysis
from visualization import (plot_time_series, plot_phase_portrait, 
                          plot_bifurcation, plot_feedback_comparison)

def main():
    """
    Run complete analysis pipeline
    """
    print("=" * 60)
    print("PERMAFROST-CARBON-CLIMATE MODEL")
    print("=" * 60)
    print()
    
    # Setup
    params = get_params_dict()
    initial_conditions = get_initial_conditions()
    
    print("Initial conditions:")
    print(f"  C_atm = {initial_conditions[0]} Pg C")
    print(f"  C_active = {initial_conditions[1]} Pg C")
    print(f"  C_deep = {initial_conditions[2]} Pg C")
    print(f"  T = {initial_conditions[3]} K ({initial_conditions[3]-273}°C)")
    print()
    
    # ========================================
    # PHASE 1: TIME SERIES SIMULATIONS
    # ========================================
    print("=" * 60)
    print("PHASE 1: TIME SERIES SIMULATIONS")
    print("=" * 60)
    print()
    
    results = run_all_scenarios(initial_conditions, params)
    
    # Save results
    results_file = '../results/time_series_results.txt'
    save_results(results, results_file)
    print()
    
    # ========================================
    # PHASE 2: BIFURCATION ANALYSIS
    # ========================================
    print("=" * 60)
    print("PHASE 2: BIFURCATION ANALYSIS")
    print("=" * 60)
    print()
    
    E_range = (0, 30)  # 0 to 30 Pg C/year
    bifurcation_data = bifurcation_analysis(params, E_range, n_points=50)
    print()
    
    # ========================================
    # PHASE 3: FEEDBACK ANALYSIS
    # ========================================
    print("=" * 60)
    print("PHASE 3: FEEDBACK ANALYSIS")
    print("=" * 60)
    print()
    
    feedback_analysis_results = analyze_feedbacks(results)
    
    print("Feedback strengths:")
    for scenario, data in feedback_analysis_results.items():
        print(f"\n{scenario}:")
        print(f"  Temperature change: {data['delta_T']:.2f} K")
        print(f"  Carbon released: {data['carbon_released']:.1f} Pg C")
        print(f"  Permafrost loss: {data['permafrost_loss_fraction']*100:.1f}%")
    print()
    
    # Save analysis
    analysis_file = '../results/analysis_results.txt'
    save_analysis(None, bifurcation_data, feedback_analysis_results, analysis_file)
    print()
    
    # ========================================
    # PHASE 4: VISUALIZATION
    # ========================================
    print("=" * 60)
    print("PHASE 4: GENERATING FIGURES")
    print("=" * 60)
    print()
    
    # Figure 1: Time series
    plot_time_series(results, '../figures/figure1_time_series.png')
    
    # Figure 2: Phase portrait
    plot_phase_portrait(results, '../figures/figure2_phase_portrait.png')
    
    # Figure 3: Bifurcation diagram
    plot_bifurcation(bifurcation_data, '../figures/figure3_bifurcation.png')
    
    # Figure 4: Feedback comparison
    plot_feedback_comparison(feedback_analysis_results, '../figures/figure4_feedbacks.png')
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print()
    print("Results saved in: ../results/")
    print("Figures saved in: ../figures/")
    print()

if __name__ == "__main__":
    main()