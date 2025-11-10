"""
Phase 3: Complete Analysis Runner

This script runs all Phase 3 analyses:
1. Phase space analysis with nullclines
2. Equilibrium point finding and stability analysis
3. Sensitivity to initial conditions
4. Comprehensive feedback quantification
5. Generates all visualizations and tables
"""

import numpy as np
import sys

# Import our modules
import parameters as params
import model
import visualization as viz
import phase3_analysis as p3
import feedback as fb


def main():
    """
    Main execution function for Phase 3
    """
    print("="*70)
    print("ARCTIC PERMAFROST-CARBON-CLIMATE MODEL")
    print("Phase 3: Phase Space Analysis & Feedback Quantification")
    print("="*70)
    
    # Get default parameters
    params_dict = params.get_default_params()
    
    # ========================================================================
    # PART 1: PHASE SPACE ANALYSIS WITH NULLCLINES
    # ========================================================================
    print("\n" + "="*70)
    print("PART 1: PHASE SPACE ANALYSIS WITH NULLCLINES")
    print("="*70)
    
    # Analysis 1: C_atm vs T_s phase portrait
    print("\n[1/3] Creating C_atm vs T_s phase portrait with nullclines...")
    
    # Run trajectories with different initial conditions
    base_ic = [1000, 50, 600, 0]
    perturbations = [
        np.array([100, 10, 20, 1]),    # Slightly warmer start
        np.array([-100, -10, -20, -1]), # Slightly cooler start
        np.array([200, 20, 50, 2]),    # Much warmer start
    ]
    
    print("  Running simulations with varying initial conditions...")
    trajectories_catm_T = p3.test_initial_condition_sensitivity(
        params_dict, base_ic, perturbations, t_span=(0, 100), dt=0.5
    )
    
    # Create phase portrait
    print("  Creating phase portrait...")
    var_ranges = {'C_atm': (550, 750), 'T_s': (-5, 40)}
    fixed_vars = {'C_frozen': 500, 'C_active': 100}
    
    p3.plot_phase_portrait_with_nullclines(
        var_ranges, fixed_vars, params_dict,
        trajectories=trajectories_catm_T,
        title="Phase Portrait: Atmospheric CO₂ vs Temperature",
        filename="phase3_catm_vs_T_nullclines.png"
    )
    
    # Analysis 2: C_active vs T_s phase portrait
    print("\n[2/3] Creating C_active vs T_s phase portrait...")
    
    var_ranges2 = {'C_active': (0, 300), 'T_s': (-5, 40)}
    fixed_vars2 = {'C_frozen': 500, 'C_atm': 650}
    
    # Run trajectories
    print("  Running simulations...")
    trajectories_cactive_T = []
    for idx, pert in enumerate([np.array([0, 0, 0, 0])] + perturbations[:2]):
        ic = base_ic + pert
        ic = np.maximum(ic, 0)
        t, sol = model.run_simulation((0, 100), y0=ic, params_dict=params_dict, dt=0.5)
        trajectories_cactive_T.append((t, sol))
    
    p3.plot_phase_portrait_with_nullclines(
        var_ranges2, fixed_vars2, params_dict,
        trajectories=trajectories_cactive_T,
        title="Phase Portrait: Active Carbon vs Temperature",
        filename="phase3_cactive_vs_T.png"
    )
    
    print("\n✓ Phase space analysis complete!")
    
    # ========================================================================
    # PART 2: EQUILIBRIUM AND STABILITY ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("PART 2: EQUILIBRIUM POINT AND STABILITY ANALYSIS")
    print("="*70)
    
    print("\n[1/2] Finding equilibrium points...")
    
    # Try to find equilibrium with multiple guesses
    equilibrium_found = False
    equilibrium = None
    
    # Try long simulation to see where system goes
    print("  Running long-term simulation to identify equilibrium...")
    t_long, sol_long = model.run_simulation((0, 1000), params_dict=params_dict, dt=2.0)
    
    # Check if system has equilibrated
    final_rates = model.permafrost_model(sol_long[-1, :], 1000, params_dict)
    max_rate = np.max(np.abs(final_rates))
    
    if max_rate < 0.01:
        equilibrium = sol_long[-1, :]
        equilibrium_found = True
        print(f"  ✓ System reached equilibrium (max |dY/dt| = {max_rate:.2e})")
    else:
        print(f"  ⚠ System has not fully equilibrated (max |dY/dt| = {max_rate:.2e})")
        # Use final state as approximate equilibrium
        equilibrium = sol_long[-1, :]
        print("  Using final state as approximate equilibrium for analysis")
    
    # Analyze stability
    print("\n[2/2] Performing stability analysis...")
    analysis = p3.analyze_stability(equilibrium, params_dict, t=500)
    
    # Print detailed analysis
    p3.print_equilibrium_analysis(equilibrium, analysis)
    
    # Create stability visualization
    print("\nCreating stability analysis visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Equilibrium and Stability Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Eigenvalue spectrum
    ax = axes[0, 0]
    eigenvals = analysis['eigenvalues']
    ax.scatter(eigenvals.real, eigenvals.imag, s=200, c='red', 
              marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('Eigenvalue Spectrum', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add labels
    for i, ev in enumerate(eigenvals):
        ax.annotate(f'λ{i+1}', (ev.real, ev.imag), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Plot 2: Jacobian heatmap
    ax = axes[0, 1]
    J = analysis['jacobian']
    im = ax.imshow(J, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['C_frozen', 'C_active', 'C_atm', 'T_s'])
    ax.set_yticklabels(['dC_f/dt', 'dC_a/dt', 'dC_atm/dt', 'dT_s/dt'])
    ax.set_title('Jacobian Matrix', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='∂f/∂y')
    
    # Plot 3: Long-term trajectory to equilibrium
    ax = axes[1, 0]
    ax.plot(t_long, sol_long[:, 3], 'r-', linewidth=2, label='Temperature')
    ax.set_xlabel('Time [years]', fontsize=12)
    ax.set_ylabel('Temperature [°C]', fontsize=12)
    ax.set_title('Approach to Equilibrium', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Stability classification
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create text summary
    summary_text = f"""
    STABILITY CLASSIFICATION
    
    Status: {analysis['stability']}
    
    Equilibrium State:
      • C_frozen = {equilibrium[0]:.1f} PgC
      • C_active = {equilibrium[1]:.1f} PgC
      • C_atm = {equilibrium[2]:.1f} PgC
      • T_s = {equilibrium[3]:.1f} °C
    
    Eigenvalue Summary:
      • λ1 = {eigenvals[0].real:.3f} + {eigenvals[0].imag:.3f}i
      • λ2 = {eigenvals[1].real:.3f} + {eigenvals[1].imag:.3f}i
      • λ3 = {eigenvals[2].real:.3f} + {eigenvals[2].imag:.3f}i
      • λ4 = {eigenvals[3].real:.3f} + {eigenvals[3].imag:.3f}i
    
    Jacobian Properties:
      • Trace = {analysis['trace']:.3f}
      • Det = {analysis['determinant']:.3f}
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig("phase3_stability_analysis.png", 
               dpi=300, bbox_inches='tight')
    print("Figure saved: phase3_stability_analysis.png")
    
    plt.close()
    
    print("\n✓ Equilibrium and stability analysis complete!")
    
    # ========================================================================
    # PART 3: COMPREHENSIVE FEEDBACK QUANTIFICATION
    # ========================================================================
    print("\n" + "="*70)
    print("PART 3: COMPREHENSIVE FEEDBACK QUANTIFICATION")
    print("="*70)
    
    print("\n[1/4] Generating feedback scenarios...")
    scenarios = fb.generate_key_feedback_scenarios()
    print(f"  Created {len(scenarios)} scenarios")
    
    print("\n[2/4] Running all feedback scenarios...")
    results = fb.quantify_feedbacks(scenarios, t_span=(0, 100), dt=0.5)
    
    print("\n[3/4] Calculating feedback contributions...")
    contributions = fb.calculate_feedback_contributions(results)
    
    # Display results
    print("\n" + "-"*70)
    print("FEEDBACK QUANTIFICATION RESULTS")
    print("-"*70)
    
    print("\nFinal States (Year 100):")
    print(results['final_states'][['T_s', 'C_atm', 'C_frozen']])
    
    print("\n" + "-"*70)
    print("FEEDBACK REMOVAL EFFECTS")
    print("(How much does removing each feedback reduce warming?)")
    print("-"*70)
    if contributions['removal_effects'] is not None:
        print(contributions['removal_effects'][['ΔT_when_removed', 'T_contribution_%']])
    
    print("\n" + "-"*70)
    print("INDIVIDUAL FEEDBACK EFFECTS")
    print("(How much does each feedback alone contribute?)")
    print("-"*70)
    if contributions['individual_effects'] is not None:
        print(contributions['individual_effects'][['ΔT_alone', 'final_T_alone']])
    
    print("\n[4/4] Creating visualizations and tables...")
    
    # Create comparison table
    table = fb.create_comparison_table(results, contributions,
                                       filename="phase3_feedback_table.csv")
    print("\nComparison Table:")
    print(table)
    
    # Create visualization: bar chart
    fb.plot_feedback_comparison(results, variable='T_s',
                                filename="phase3_feedback_bars.png")
    
    # Create visualization: contribution analysis
    fb.plot_contribution_analysis(contributions,
                                  filename="phase3_contributions.png")
    
    # Create visualization: time series comparison
    print("\nCreating time series comparison...")
    viz.plot_comparison_scenarios(
        results['simulations'],
        title="Feedback Scenario Time Series Comparison",
        filename="phase3_feedback_timeseries.png"
    )
    
    print("\n✓ Feedback quantification complete!")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE: KEY FINDINGS")
    print("="*70)
    
    # Extract key findings
    all_fb_T = results['final_states'].loc['All Feedbacks', 'T_s']
    no_fb_T = results['final_states'].loc['No Feedbacks', 'T_s']
    no_albedo_T = results['final_states'].loc['No Albedo', 'T_s']
    only_albedo_T = results['final_states'].loc['Only Albedo', 'T_s']
    
    print("\n✓ Key Findings:")
    print(f"  1. With all feedbacks: T = {all_fb_T:.1f}°C")
    print(f"  2. With no feedbacks: T = {no_fb_T:.1f}°C")
    print(f"  3. Without albedo feedback: T = {no_albedo_T:.1f}°C")
    print(f"  4. With only albedo feedback: T = {only_albedo_T:.1f}°C")
    
    albedo_importance = all_fb_T - no_albedo_T
    print(f"\n  ⭐ Albedo feedback contributes {albedo_importance:.1f}°C " +
          f"({100*albedo_importance/all_fb_T:.1f}% of total warming)")
    
    print("\n✓ Equilibrium Analysis:")
    print(f"  System reaches: T = {equilibrium[3]:.1f}°C, C_atm = {equilibrium[2]:.1f} PgC")
    print(f"  Stability: {analysis['stability']}")
    
    print("\n✓ Generated Outputs:")
    print("  - phase3_catm_vs_T_nullclines.png")
    print("  - phase3_cactive_vs_T.png")
    print("  - phase3_stability_analysis.png")
    print("  - phase3_feedback_bars.png")
    print("  - phase3_contributions.png")
    print("  - phase3_feedback_timeseries.png")
    print("  - phase3_feedback_table.csv")
    
    print("\n" + "="*70)
    print("PHASE 3 ANALYSIS COMPLETE!")
    print("="*70)
    
    return 0


# ============================================================================
# RUN MAIN
# ============================================================================

if __name__ == "__main__":
    # Need matplotlib for saving figures
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)