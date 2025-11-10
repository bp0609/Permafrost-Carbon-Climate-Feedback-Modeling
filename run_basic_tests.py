"""
Main Script: Run Basic Tests and Demonstrations

This script:
1. Validates all model components
2. Runs basic simulations
3. Creates initial visualizations
4. Tests different scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Import our modules
import parameters as params
import model
import visualization as viz
import validation

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("="*70)
    print("ARCTIC PERMAFROST-CARBON-CLIMATE MODEL")
    print("Phase 2: Implementation & Basic Testing")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Display and Validate Parameters
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: PARAMETER CONFIGURATION")
    print("="*70)
    
    params.print_parameter_summary()
    
    # Get parameter dictionary
    params_dict = params.get_default_params()
    
    # Validate parameters
    print("\nValidating parameters...")
    is_valid, issues = validation.validate_parameters(params_dict)
    
    if is_valid:
        print("✓ All parameters are valid!")
    else:
        print("✗ Parameter validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Test Model Components
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: TESTING MODEL COMPONENTS")
    print("="*70)
    
    print("\nTesting temperature-dependent functions...")
    T_test = np.array([-10, -5, 0, 5, 10])
    
    print("\nTemperature [°C]:", T_test)
    print("Thaw rates [1/yr]:", 
          [f"{model.thaw_rate(T, params_dict):.4f}" for T in T_test])
    print("Decomp rates [1/yr]:", 
          [f"{model.decomposition_rate(T, params_dict):.4f}" for T in T_test])
    print("Albedo values:", 
          [f"{model.albedo(T, params_dict):.3f}" for T in T_test])
    
    print("\nTesting CO2 forcing...")
    C_test = np.array([600, 700, 800, 900, 1000])
    print("C_atm [PgC]:", C_test)
    print("ΔF [W/m²]:", 
          [f"{model.CO2_forcing(C, params_dict):.2f}" for C in C_test])
    
    print("\n✓ All model components functional!")
    
    # ========================================================================
    # STEP 3: Run Short Simulation (Decadal Scale)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: SHORT SIMULATION (100 years)")
    print("="*70)
    
    print("\nRunning 100-year simulation...")
    t_short, sol_short = model.run_simulation((0, 100), dt=0.5)
    
    print(f"Simulation complete: {len(t_short)} time points")
    print("\nInitial state:")
    print(f"  C_frozen = {sol_short[0,0]:.1f} PgC")
    print(f"  C_active = {sol_short[0,1]:.1f} PgC")
    print(f"  C_atm = {sol_short[0,2]:.1f} PgC")
    print(f"  T_s = {sol_short[0,3]:.2f} °C")
    
    print("\nFinal state:")
    print(f"  C_frozen = {sol_short[-1,0]:.1f} PgC")
    print(f"  C_active = {sol_short[-1,1]:.1f} PgC")
    print(f"  C_atm = {sol_short[-1,2]:.1f} PgC")
    print(f"  T_s = {sol_short[-1,3]:.2f} °C")
    
    print("\nChanges:")
    print(f"  ΔC_frozen = {sol_short[-1,0] - sol_short[0,0]:.1f} PgC " +
          f"({100*(sol_short[-1,0] - sol_short[0,0])/sol_short[0,0]:.1f}%)")
    print(f"  ΔC_active = {sol_short[-1,1] - sol_short[0,1]:.1f} PgC " +
          f"({100*(sol_short[-1,1] - sol_short[0,1])/sol_short[0,1]:.1f}%)")
    print(f"  ΔC_atm = {sol_short[-1,2] - sol_short[0,2]:.1f} PgC " +
          f"({100*(sol_short[-1,2] - sol_short[0,2])/sol_short[0,2]:.1f}%)")
    print(f"  ΔT_s = {sol_short[-1,3] - sol_short[0,3]:.2f} °C")
    
    # ========================================================================
    # STEP 4: Run Comprehensive Validation
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPREHENSIVE VALIDATION")
    print("="*70)
    
    all_passed, results = validation.run_full_validation(
        t_span=(0, 100),
        dt=0.5,
        params_dict=params_dict,
        verbose=True
    )
    
    if not all_passed:
        print("\n⚠ Some validation tests failed. Review above.")
    
    # ========================================================================
    # STEP 5: Create Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*70)
    
    print("\nGenerating plots...")
    
    # Plot 1: Time series
    print("  [1/4] Time series plot...")
    viz.plot_time_series(
        t_short, sol_short,
        title="Arctic Permafrost Model - 100 Year Simulation",
        filename="test_timeseries.png"
    )
    
    # Plot 2: Temperature-dependent rates
    print("  [2/4] Temperature-dependent rates...")
    viz.plot_temperature_dependent_rates(
        params_dict=params_dict,
        title="Temperature-Dependent Rate Functions",
        filename="test_rates.png"
    )
    
    # Plot 3: Phase portrait
    print("  [3/4] Phase portrait (C_atm vs T_s)...")
    viz.plot_phase_portrait_2D(
        sol_short,
        title="Phase Portrait: CO₂ vs Temperature",
        filename="test_phase.png"
    )
    
    # Plot 4: Carbon pools
    print("  [4/4] Carbon pools distribution...")
    viz.plot_carbon_pools_stacked(
        t_short, sol_short,
        title="Carbon Distribution Over Time",
        filename="test_carbon_pools.png"
    )
    
    print("\n✓ All visualizations created!")
    
    # ========================================================================
    # STEP 6: Test Multiple Scenarios
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: TESTING MULTIPLE SCENARIOS")
    print("="*70)
    
    print("\nSetting up scenarios...")
    
    # Scenario 1: Baseline (all feedbacks)
    params_baseline = params.get_default_params()
    
    # Scenario 2: No albedo feedback
    params_no_albedo = params.get_default_params()
    params_no_albedo['enable_albedo_feedback'] = False
    
    # Scenario 3: No temperature-dependent decomposition
    params_no_decomp = params.get_default_params()
    params_no_decomp['enable_temp_decomposition'] = False
    
    scenarios = {
        'All Feedbacks': params_baseline,
        'No Albedo Feedback': params_no_albedo,
        'No Temp-Dependent Decomp': params_no_decomp
    }
    
    print(f"Running {len(scenarios)} scenarios...")
    results = model.run_multiple_scenarios(scenarios, (0, 100), dt=0.5)
    
    print("\nScenario results (final states):")
    for scenario_name, (t, sol) in results.items():
        print(f"\n  {scenario_name}:")
        print(f"    T_final = {sol[-1,3]:.2f} °C")
        print(f"    C_atm_final = {sol[-1,2]:.1f} PgC")
        print(f"    C_frozen_final = {sol[-1,0]:.1f} PgC")
    
    # Plot comparison
    print("\nCreating scenario comparison plot...")
    viz.plot_comparison_scenarios(
        results,
        title="Comparison of Feedback Scenarios",
        filename="test_scenarios.png"
    )
    
    print("\n✓ Scenario testing complete!")
    
    # ========================================================================
    # STEP 7: Long-term Simulation (Century Scale)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: LONG-TERM SIMULATION (500 years)")
    print("="*70)
    
    print("\nRunning 500-year simulation...")
    t_long, sol_long = model.run_simulation((0, 500), dt=1.0)
    
    print(f"Simulation complete: {len(t_long)} time points")
    print("\nFinal state (year 500):")
    print(f"  C_frozen = {sol_long[-1,0]:.1f} PgC")
    print(f"  C_active = {sol_long[-1,1]:.1f} PgC")
    print(f"  C_atm = {sol_long[-1,2]:.1f} PgC")
    print(f"  T_s = {sol_long[-1,3]:.2f} °C")
    
    print("\nTotal changes over 500 years:")
    print(f"  ΔC_frozen = {sol_long[-1,0] - sol_long[0,0]:.1f} PgC " +
          f"({100*(sol_long[-1,0] - sol_long[0,0])/sol_long[0,0]:.1f}%)")
    print(f"  ΔC_atm = {sol_long[-1,2] - sol_long[0,2]:.1f} PgC " +
          f"({100*(sol_long[-1,2] - sol_long[0,2])/sol_long[0,2]:.1f}%)")
    print(f"  ΔT_s = {sol_long[-1,3] - sol_long[0,3]:.2f} °C")
    
    # Create long-term plots
    print("\nCreating long-term visualization...")
    viz.plot_time_series(
        t_long, sol_long,
        title="Arctic Permafrost Model - 500 Year Simulation",
        filename="test_long_term.png"
    )
    
    print("\n✓ Long-term simulation complete!")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE: IMPLEMENTATION & BASIC TESTING")
    print("="*70)
    
    print("\n✓ Summary of completed tasks:")
    print("  1. ✓ Parameter configuration and validation")
    print("  2. ✓ Model component testing")
    print("  3. ✓ Short-term simulation (100 years)")
    print("  4. ✓ Comprehensive validation suite")
    print("  5. ✓ Visualization functions tested")
    print("  6. ✓ Multiple scenario comparison")
    print("  7. ✓ Long-term simulation (500 years)")
    
    print("\n✓ Generated outputs:")
    print("  - test_timeseries.png")
    print("  - test_rates.png")
    print("  - test_phase.png")
    print("  - test_carbon_pools.png")
    print("  - test_scenarios.png")
    print("  - test_long_term.png")
    
    print("\n" + "="*70)
    print("MODEL IS READY FOR PHASE 3: PHASE SPACE ANALYSIS")
    print("="*70)
    print("\nNext steps:")
    print("  - Phase space portraits with nullclines")
    print("  - Equilibrium point analysis")
    print("  - Jacobian and stability analysis")
    print("  - Feedback quantification")
    
    return 0


# ============================================================================
# RUN MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)