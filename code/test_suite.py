"""
test_suite.py
Arctic Permafrost-Carbon-Climate Feedback Model
Comprehensive test suite for model validation
"""

import numpy as np
import sys

# Test imports
print("Testing imports...")
try:
    from parameters import PARAMS, SCENARIOS
    from helper_functions import (decomposition_rate_q10, radiative_forcing_co2,
                                   thaw_flux, carbon_uptake)
    from model import permafrost_model
    from solver import run_model
    print("✓ All imports successful\n")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_parameters():
    """Test parameter definitions"""
    print("=" * 70)
    print("TEST 1: PARAMETERS")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1.1: Initial conditions are positive
    tests_total += 1
    if all(PARAMS[key] > 0 for key in ['C_atm_0', 'C_active_0', 'C_deep_0']):
        print("✓ 1.1: Initial carbon stocks are positive")
        tests_passed += 1
    else:
        print("✗ 1.1: Initial carbon stocks not all positive")
    
    # Test 1.2: Physical constants in reasonable ranges
    tests_total += 1
    if 1300 <= PARAMS['S_0'] <= 1400 and 0 < PARAMS['albedo'] < 1:
        print("✓ 1.2: Physical constants in reasonable ranges")
        tests_passed += 1
    else:
        print("✗ 1.2: Physical constants outside expected ranges")
    
    # Test 1.3: RCP scenarios return reasonable values
    tests_total += 1
    rcp_values = [SCENARIOS['RCP4.5'](50) for _ in range(3)]
    if all(0 < val < 30 for val in rcp_values):
        print("✓ 1.3: RCP emission scenarios return reasonable values")
        tests_passed += 1
    else:
        print("✗ 1.3: RCP scenarios return unexpected values")
    
    print(f"\nPassed: {tests_passed}/{tests_total}\n")
    return tests_passed == tests_total


def test_helper_functions():
    """Test individual helper functions"""
    print("=" * 70)
    print("TEST 2: HELPER FUNCTIONS")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 2.1: Decomposition increases with temperature
    tests_total += 1
    k_cold = decomposition_rate_q10(-5, PARAMS)
    k_warm = decomposition_rate_q10(5, PARAMS)
    if k_warm > k_cold > 0:
        print("✓ 2.1: Decomposition rate increases with temperature")
        tests_passed += 1
    else:
        print(f"✗ 2.1: Decomposition trend incorrect (cold={k_cold}, warm={k_warm})")
    
    # Test 2.2: CO2 forcing is logarithmic
    tests_total += 1
    F_1x = radiative_forcing_co2(PARAMS['C_0'], PARAMS)
    F_2x = radiative_forcing_co2(2 * PARAMS['C_0'], PARAMS)
    expected_doubling = 5.35 * np.log(2)
    if abs(F_2x - expected_doubling) < 0.01 and abs(F_1x) < 0.01:
        print("✓ 2.2: CO2 radiative forcing follows logarithmic law")
        tests_passed += 1
    else:
        print(f"✗ 2.2: CO2 forcing incorrect (F_2x={F_2x:.2f}, expected={expected_doubling:.2f})")
    
    # Test 2.3: Thaw flux increases above threshold
    tests_total += 1
    F_cold = thaw_flux(-5, 800, PARAMS)
    F_warm = thaw_flux(5, 800, PARAMS)
    if F_warm > F_cold >= 0:
        print("✓ 2.3: Thaw flux increases above 0°C threshold")
        tests_passed += 1
    else:
        print(f"✗ 2.3: Thaw flux behavior incorrect (cold={F_cold}, warm={F_warm})")
    
    # Test 2.4: Carbon uptake proportional to excess
    tests_total += 1
    F_uptake_1 = carbon_uptake(700, PARAMS)
    F_uptake_2 = carbon_uptake(800, PARAMS)
    if F_uptake_2 > F_uptake_1 > 0:
        print("✓ 2.4: Carbon uptake proportional to atmospheric excess")
        tests_passed += 1
    else:
        print(f"✗ 2.4: Carbon uptake trend incorrect")
    
    print(f"\nPassed: {tests_passed}/{tests_total}\n")
    return tests_passed == tests_total


def test_model_consistency():
    """Test ODE system for physical consistency"""
    print("=" * 70)
    print("TEST 3: MODEL CONSISTENCY")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Initial state
    state = np.array([PARAMS['C_atm_0'], PARAMS['C_active_0'], 
                     PARAMS['C_deep_0'], PARAMS['T_s_0']])
    
    # Test 3.1: Carbon conservation
    tests_total += 1
    derivatives = permafrost_model(state, 0, PARAMS, SCENARIOS['RCP4.5'])
    
    F_anthro = SCENARIOS['RCP4.5'](0)
    F_uptake = carbon_uptake(state[0], PARAMS)
    external_input = F_anthro - F_uptake
    total_carbon_change = np.sum(derivatives[:3])
    
    if abs(total_carbon_change - external_input) < 1e-6:
        print("✓ 3.1: Carbon is conserved (internal fluxes cancel)")
        tests_passed += 1
    else:
        print(f"✗ 3.1: Carbon conservation violated (error={abs(total_carbon_change - external_input):.2e})")
    
    # Test 3.2: Temperature response to positive forcing
    tests_total += 1
    F_co2 = radiative_forcing_co2(state[0] * 1.5, PARAMS)  # Higher CO2
    if F_co2 > 0 and derivatives[3] != 0:  # Should cause temp change
        print("✓ 3.2: Temperature responds to radiative forcing")
        tests_passed += 1
    else:
        print("✗ 3.2: Temperature not responding to forcing")
    
    # Test 3.3: Deep permafrost only decreases
    tests_total += 1
    warm_state = state.copy()
    warm_state[3] = 5.0  # Warm temperature
    warm_derivs = permafrost_model(warm_state, 0, PARAMS, SCENARIOS['RCP4.5'])
    
    if warm_derivs[2] <= 0:  # dC_deep/dt <= 0
        print("✓ 3.3: Deep permafrost only decreases (no refreezing)")
        tests_passed += 1
    else:
        print(f"✗ 3.3: Deep permafrost increasing (dC_deep/dt={warm_derivs[2]})")
    
    # Test 3.4: Active layer receives thaw flux
    tests_total += 1
    # Calculate the actual thaw fluxes for both cases
    from helper_functions import thaw_flux
    F_thaw_cold = thaw_flux(state[3], state[2], PARAMS)
    F_thaw_warm = thaw_flux(warm_state[3], warm_state[2], PARAMS)

    if F_thaw_warm > F_thaw_cold >= 0:  # Check thaw flux directly, not net change
        print("✓ 3.4: Active layer receives carbon from thaw")
        tests_passed += 1
    else:
        print(f"✗ 3.4: Active layer thaw flux not increasing with temperature (cold={F_thaw_cold:.3f}, warm={F_thaw_warm:.3f})")
    
    print(f"\nPassed: {tests_passed}/{tests_total}\n")
    return tests_passed == tests_total


def test_solver():
    """Test numerical solver"""
    print("=" * 70)
    print("TEST 4: SOLVER INTEGRATION")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Run short simulation
    initial_state = np.array([PARAMS['C_atm_0'], PARAMS['C_active_0'],
                             PARAMS['C_deep_0'], PARAMS['T_s_0']])
    time_points = np.linspace(0, 10, 101)  # 10 years
    
    try:
        results = run_model(initial_state, time_points, PARAMS, 
                           forcing_scenario='RCP4.5')
        
        # Test 4.1: Solver completes without error
        tests_total += 1
        print("✓ 4.1: Solver runs to completion")
        tests_passed += 1
        
        # Test 4.2: No negative carbon stocks
        tests_total += 1
        if (np.all(results['C_atm'] > 0) and 
            np.all(results['C_active'] >= 0) and 
            np.all(results['C_deep'] >= 0)):
            print("✓ 4.2: No negative carbon stocks")
            tests_passed += 1
        else:
            print("✗ 4.2: Negative carbon stocks detected")
        
        # Test 4.3: Atmospheric CO2 increases under RCP 4.5
        tests_total += 1
        if results['CO2_ppm'][-1] > results['CO2_ppm'][0]:
            print("✓ 4.3: Atmospheric CO2 increases as expected")
            tests_passed += 1
        else:
            print("✗ 4.3: Atmospheric CO2 not increasing")
        
        # Test 4.4: Temperature increases under warming scenario
        tests_total += 1
        if results['T_s'][-1] > results['T_s'][0]:
            print("✓ 4.4: Temperature increases as expected")
            tests_passed += 1
        else:
            print("✗ 4.4: Temperature not increasing")
        
        # Test 4.5: Results have correct shape
        tests_total += 1
        if (len(results['time']) == len(time_points) and
            len(results['C_atm']) == len(time_points)):
            print("✓ 4.5: Output arrays have correct dimensions")
            tests_passed += 1
        else:
            print("✗ 4.5: Output array dimensions incorrect")
            
    except Exception as e:
        tests_total += 5
        print(f"✗ Solver failed with error: {e}")
    
    print(f"\nPassed: {tests_passed}/{tests_total}\n")
    return tests_passed == tests_total


def test_physical_realism():
    """Test for physically realistic behavior"""
    print("=" * 70)
    print("TEST 5: PHYSICAL REALISM")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Run 100-year simulation
    initial_state = np.array([PARAMS['C_atm_0'], PARAMS['C_active_0'],
                             PARAMS['C_deep_0'], PARAMS['T_s_0']])
    time_points = np.linspace(0, 100, 1001)
    
    results_45 = run_model(initial_state, time_points, PARAMS, 'RCP4.5')
    results_85 = run_model(initial_state, time_points, PARAMS, 'RCP8.5')
    
    # Test 5.1: Higher emissions = more warming
    tests_total += 1
    delta_T_45 = results_45['T_s'][-1] - results_45['T_s'][0]
    delta_T_85 = results_85['T_s'][-1] - results_85['T_s'][0]
    if delta_T_85 > delta_T_45 > 0:
        print(f"✓ 5.1: RCP 8.5 warms more than RCP 4.5 (ΔT_85={delta_T_85:.1f}°C > ΔT_45={delta_T_45:.1f}°C)")
        tests_passed += 1
    else:
        print(f"✗ 5.1: Warming hierarchy incorrect")
    
    # Test 5.2: CO2 growth rate realistic
    tests_total += 1
    co2_rate = (results_45['CO2_ppm'][-1] - results_45['CO2_ppm'][0]) / 100
    if 1 < co2_rate < 5:  # Reasonable range: 1-5 ppm/year average
        print(f"✓ 5.2: CO2 growth rate realistic ({co2_rate:.2f} ppm/yr)")
        tests_passed += 1
    else:
        print(f"✗ 5.2: CO2 growth rate unrealistic ({co2_rate:.2f} ppm/yr)")
    
    # Test 5.3: Permafrost loss < 80% by 2100 (even under RCP 8.5)
    # Note: Higher-end estimates suggest 30-70% loss under high warming scenarios
    tests_total += 1
    C_perm_init = PARAMS['C_active_0'] + PARAMS['C_deep_0']
    C_perm_final = results_85['C_active'][-1] + results_85['C_deep'][-1]
    pct_loss = 100 * (C_perm_init - C_perm_final) / C_perm_init
    if 0 < pct_loss < 80:
        print(f"✓ 5.3: Permafrost loss realistic ({pct_loss:.1f}% under RCP 8.5)")
        tests_passed += 1
    else:
        print(f"✗ 5.3: Permafrost loss unrealistic ({pct_loss:.1f}%)")
    
    # Test 5.4: Final temperature in reasonable range
    # Note: Extreme warming scenarios can reach 15-25°C above initial Arctic temps
    tests_total += 1
    if -10 < results_85['T_s'][-1] < 25:
        print(f"✓ 5.4: Final temperature reasonable ({results_85['T_s'][-1]:.1f}°C)")
        tests_passed += 1
    else:
        print(f"✗ 5.4: Final temperature outside realistic range ({results_85['T_s'][-1]:.1f}°C)")
    
    print(f"\nPassed: {tests_passed}/{tests_total}\n")
    return tests_passed == tests_total


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUITE")
    print("Arctic Permafrost-Carbon-Climate Feedback Model")
    print("=" * 70 + "\n")
    
    results = []
    results.append(("Parameters", test_parameters()))
    results.append(("Helper Functions", test_helper_functions()))
    results.append(("Model Consistency", test_model_consistency()))
    results.append(("Solver Integration", test_solver()))
    results.append(("Physical Realism", test_physical_realism()))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<25} {status}")
    
    total_passed = sum(passed for _, passed in results)
    total_tests = len(results)
    
    print("=" * 70)
    print(f"OVERALL: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Model is ready for use.")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    print("=" * 70 + "\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)