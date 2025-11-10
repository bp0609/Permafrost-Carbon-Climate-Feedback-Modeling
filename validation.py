"""
Validation and Testing Module for Arctic Permafrost-Carbon-Climate Model

Comprehensive tests to ensure:
1. Parameter validity
2. Physical constraints
3. Conservation laws
4. Numerical stability
5. Equilibrium behavior
"""

import numpy as np
import model
import parameters as params


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_parameters(params_dict):
    """
    Check that all parameters are physically reasonable
    
    Parameters:
    -----------
    params_dict : dict
        Parameter dictionary to validate
    
    Returns:
    --------
    is_valid : bool
        True if all parameters are valid
    issues : list
        List of issues found (empty if valid)
    """
    issues = []
    
    # Check positive values
    positive_params = [
        'k_thaw_base', 'k_decomp_base', 'Q10_decomp', 
        'tau_atm_ocean', 'C_heat', 'lambda_climate',
        'C_frozen_init', 'C_active_init', 'CO2_preind_PgC'
    ]
    
    for param in positive_params:
        if params_dict[param] <= 0:
            issues.append(f"{param} must be positive, got {params_dict[param]}")
    
    # Check albedo range
    if not (0 <= params_dict['alpha_min'] <= 1):
        issues.append(f"alpha_min must be in [0,1], got {params_dict['alpha_min']}")
    if not (0 <= params_dict['alpha_max'] <= 1):
        issues.append(f"alpha_max must be in [0,1], got {params_dict['alpha_max']}")
    if params_dict['alpha_min'] >= params_dict['alpha_max']:
        issues.append("alpha_min must be less than alpha_max")
    
    # Check Q10 is reasonable (typically 1.5 to 3)
    if not (1.0 <= params_dict['Q10_decomp'] <= 5.0):
        issues.append(f"Q10_decomp seems unreasonable: {params_dict['Q10_decomp']}")
    
    # Check temperature sensitivity
    if params_dict['beta_thaw'] < 0:
        issues.append("beta_thaw should be positive for warming-induced thawing")
    
    # Check initial carbon stocks
    if params_dict['C_frozen_init'] < params_dict['C_active_init']:
        issues.append("Warning: Active carbon larger than frozen carbon initially")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


# ============================================================================
# PHYSICAL CONSTRAINT CHECKS
# ============================================================================

def check_non_negativity(solution, tolerance=1e-10):
    """
    Check that all state variables remain non-negative
    
    Parameters:
    -----------
    solution : array
        Solution array from model run
    tolerance : float
        Tolerance for considering values as zero
    
    Returns:
    --------
    is_valid : bool
        True if all values are non-negative
    violations : dict
        Dictionary of violations with indices
    """
    violations = {}
    
    var_names = ['C_frozen', 'C_active', 'C_atm', 'T_s']
    
    for i, name in enumerate(var_names):
        if name == 'T_s':
            # Temperature can be negative (it's in Celsius)
            continue
        
        negative_indices = np.where(solution[:, i] < -tolerance)[0]
        if len(negative_indices) > 0:
            violations[name] = {
                'indices': negative_indices,
                'min_value': solution[negative_indices, i].min()
            }
    
    is_valid = len(violations) == 0
    
    return is_valid, violations


def check_carbon_conservation(t, solution, params_dict, tolerance=50.0):
    """
    Check if total carbon is approximately conserved
    (accounting for anthropogenic emissions and ocean uptake)
    
    Parameters:
    -----------
    t : array
        Time array
    solution : array
        Solution array
    params_dict : dict
        Parameter dictionary
    tolerance : float
        Tolerance for conservation check [PgC]
    
    Returns:
    --------
    is_conserved : bool
        True if carbon is approximately conserved
    report : dict
        Detailed report on carbon balance
    """
    C_frozen = solution[:, 0]
    C_active = solution[:, 1]
    C_atm = solution[:, 2]
    
    # Total carbon in system
    C_total = C_frozen + C_active + C_atm
    
    # Initial total carbon
    C_initial = C_total[0]
    
    # Calculate expected change from emissions and ocean uptake
    # Integrate emissions and ocean uptake over time for accurate accounting
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    
    # Calculate emissions at each time point and integrate
    E_2020 = params_dict['E_2020']
    r = params_dict['E_growth_rate']
    emissions_at_t = E_2020 * np.exp(r * t)
    C_added_emissions = np.trapezoid(emissions_at_t, t)
    
    # Integrate ocean uptake over time
    # Ocean uptake is based on EXCESS CO2 above pre-industrial
    tau = params_dict['tau_atm_ocean']
    C_excess = np.maximum(C_atm - params_dict['CO2_preind_PgC'], 0)
    C_removed_ocean = np.trapezoid(C_excess / tau, t)  # Integrated flux
    
    # Expected final total
    C_expected_final = C_initial + C_added_emissions - C_removed_ocean
    
    # Actual final total
    C_final = C_total[-1]
    
    # Check conservation
    difference = abs(C_final - C_expected_final)
    is_conserved = difference < tolerance
    
    report = {
        'C_initial': C_initial,
        'C_final': C_final,
        'C_expected_final': C_expected_final,
        'difference': difference,
        'relative_change': (C_final - C_initial) / C_initial,
        'added_from_emissions': C_added_emissions,
        'removed_to_ocean': C_removed_ocean,
        'is_conserved': is_conserved
    }
    
    return is_conserved, report


def check_physical_reasonableness(solution, params_dict):
    """
    Check if solutions are physically reasonable
    
    Parameters:
    -----------
    solution : array
        Solution array
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    is_reasonable : bool
        True if all values are reasonable
    warnings : list
        List of warnings about unreasonable values
    """
    warnings = []
    
    C_frozen = solution[:, 0]
    C_active = solution[:, 1]
    C_atm = solution[:, 2]
    T_s = solution[:, 3]
    
    # Check frozen carbon doesn't increase (only thaws)
    if np.any(np.diff(C_frozen) > 1e-6):
        warnings.append("Frozen carbon increased (should only decrease)")
    
    # Check temperature range
    if np.any(T_s < -50):
        warnings.append(f"Temperature too cold: {T_s.min():.1f}°C")
    if np.any(T_s > 50):
        warnings.append(f"Temperature too hot: {T_s.max():.1f}°C")
    
    # Check CO2 doesn't drop below pre-industrial
    if np.any(C_atm < params_dict['CO2_preind_PgC'] * 0.5):
        warnings.append("Atmospheric CO2 dropped far below pre-industrial")
    
    # Check CO2 doesn't exceed extreme values
    if np.any(C_atm > 3000):
        warnings.append(f"Atmospheric CO2 very high: {C_atm.max():.1f} PgC")
    
    # Check active layer doesn't exceed initial frozen pool
    if np.any(C_active > params_dict['C_frozen_init'] * 1.5):
        warnings.append("Active layer carbon exceeds reasonable bounds")
    
    is_reasonable = len(warnings) == 0
    
    return is_reasonable, warnings


# ============================================================================
# NUMERICAL STABILITY CHECKS
# ============================================================================

def check_numerical_stability(solution, max_relative_change=2.0):
    """
    Check for numerical instabilities (e.g., oscillations, blowup)
    
    Parameters:
    -----------
    solution : array
        Solution array
    max_relative_change : float
        Maximum allowed relative change between time steps
    
    Returns:
    --------
    is_stable : bool
        True if solution is stable
    issues : list
        List of stability issues found
    """
    issues = []
    
    var_names = ['C_frozen', 'C_active', 'C_atm', 'T_s']
    
    for i, name in enumerate(var_names):
        var = solution[:, i]
        
        # Check for NaN or Inf
        if np.any(~np.isfinite(var)):
            issues.append(f"{name} contains NaN or Inf")
            continue
        
        # Check for excessive oscillations
        if len(var) > 2:
            changes = np.diff(var)
            sign_changes = np.sum(np.diff(np.sign(changes)) != 0)
            if sign_changes > len(var) * 0.3:  # More than 30% sign changes
                issues.append(f"{name} shows excessive oscillations")
        
        # Check for large relative changes (skip near-zero values)
        if len(var) > 1:
            # Use absolute threshold to avoid issues with near-zero values
            abs_threshold = 1e-3 if name == 'T_s' else 1.0
            
            # Only check relative changes where values are significant
            mask = np.abs(var[:-1]) > abs_threshold
            if np.any(mask):
                relative_changes = np.abs(np.diff(var)[mask]) / (np.abs(var[:-1][mask]) + 1e-10)
                if np.any(relative_changes > max_relative_change):
                    max_change = relative_changes.max()
                    issues.append(f"{name} has large relative change: {max_change:.2f}")
    
    is_stable = len(issues) == 0
    
    return is_stable, issues


# ============================================================================
# EQUILIBRIUM TESTS
# ============================================================================

def find_equilibrium_state(solution, t, tolerance=1e-6):
    """
    Check if system reaches equilibrium and find equilibrium state
    
    Parameters:
    -----------
    solution : array
        Solution array
    t : array
        Time array
    tolerance : float
        Tolerance for considering equilibrium reached
    
    Returns:
    --------
    has_equilibrium : bool
        True if equilibrium is reached
    equilibrium_info : dict
        Information about equilibrium state
    """
    # Calculate time derivatives (rates of change)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    rates = np.gradient(solution, dt, axis=0)
    
    # Check last 10% of simulation
    n_check = max(len(t) // 10, 10)
    rates_end = rates[-n_check:, :]
    
    # Check if rates are small
    max_rates = np.max(np.abs(rates_end), axis=0)
    
    var_names = ['C_frozen', 'C_active', 'C_atm', 'T_s']
    equilibrium_reached = {}
    
    for i, name in enumerate(var_names):
        equilibrium_reached[name] = max_rates[i] < tolerance
    
    has_equilibrium = all(equilibrium_reached.values())
    
    equilibrium_info = {
        'has_equilibrium': has_equilibrium,
        'equilibrium_state': solution[-1, :] if has_equilibrium else None,
        'max_rates': dict(zip(var_names, max_rates)),
        'equilibrium_reached': equilibrium_reached
    }
    
    return has_equilibrium, equilibrium_info


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def run_full_validation(t_span=(0, 100), dt=0.5, params_dict=None, verbose=True):
    """
    Run complete validation test suite
    
    Parameters:
    -----------
    t_span : tuple
        Time span for simulation
    dt : float
        Time step
    params_dict : dict, optional
        Parameter dictionary
    verbose : bool
        Print detailed results
    
    Returns:
    --------
    all_tests_passed : bool
        True if all tests pass
    results : dict
        Detailed results from all tests
    """
    if params_dict is None:
        params_dict = params.get_default_params()
    
    results = {}
    all_tests_passed = True
    
    if verbose:
        print("="*70)
        print("RUNNING COMPREHENSIVE VALIDATION TESTS")
        print("="*70)
    
    # Test 1: Parameter Validation
    if verbose:
        print("\n[1/6] Validating parameters...")
    is_valid, issues = validate_parameters(params_dict)
    results['parameter_validation'] = {
        'passed': is_valid,
        'issues': issues
    }
    all_tests_passed &= is_valid
    
    if verbose:
        if is_valid:
            print("  ✓ All parameters valid")
        else:
            print(f"  ✗ Parameter issues found: {len(issues)}")
            for issue in issues:
                print(f"    - {issue}")
    
    # Run simulation for remaining tests
    if verbose:
        print("\n[2/6] Running simulation...")
    try:
        t, solution = model.run_simulation(t_span, params_dict=params_dict, dt=dt)
        results['simulation'] = {
            'passed': True,
            'message': 'Simulation completed successfully'
        }
        if verbose:
            print(f"  ✓ Simulation completed ({len(t)} time points)")
    except Exception as e:
        results['simulation'] = {
            'passed': False,
            'error': str(e)
        }
        all_tests_passed = False
        if verbose:
            print(f"  ✗ Simulation failed: {e}")
        return all_tests_passed, results
    
    # Test 2: Non-negativity
    if verbose:
        print("\n[3/6] Checking non-negativity constraints...")
    is_valid, violations = check_non_negativity(solution)
    results['non_negativity'] = {
        'passed': is_valid,
        'violations': violations
    }
    all_tests_passed &= is_valid
    
    if verbose:
        if is_valid:
            print("  ✓ All variables remain non-negative")
        else:
            print(f"  ✗ Negativity violations found:")
            for var, info in violations.items():
                print(f"    - {var}: min value = {info['min_value']:.6f}")
    
    # Test 3: Carbon Conservation
    if verbose:
        print("\n[4/6] Checking carbon conservation...")
    is_conserved, report = check_carbon_conservation(t, solution, params_dict)
    results['carbon_conservation'] = {
        'passed': is_conserved,
        'report': report
    }
    all_tests_passed &= is_conserved
    
    if verbose:
        if is_conserved:
            print("  ✓ Carbon approximately conserved")
        else:
            print("  ✗ Carbon conservation issue")
        print(f"    Initial total: {report['C_initial']:.1f} PgC")
        print(f"    Final total: {report['C_final']:.1f} PgC")
        print(f"    Expected final: {report['C_expected_final']:.1f} PgC")
        print(f"    Difference: {report['difference']:.1f} PgC")
    
    # Test 4: Physical Reasonableness
    if verbose:
        print("\n[5/6] Checking physical reasonableness...")
    is_reasonable, warnings = check_physical_reasonableness(solution, params_dict)
    results['physical_reasonableness'] = {
        'passed': is_reasonable,
        'warnings': warnings
    }
    # Warnings don't fail the test, but we note them
    
    if verbose:
        if is_reasonable:
            print("  ✓ All values physically reasonable")
        else:
            print(f"  ⚠ Physical reasonableness warnings:")
            for warning in warnings:
                print(f"    - {warning}")
    
    # Test 5: Numerical Stability
    if verbose:
        print("\n[6/6] Checking numerical stability...")
    is_stable, issues = check_numerical_stability(solution)
    results['numerical_stability'] = {
        'passed': is_stable,
        'issues': issues
    }
    all_tests_passed &= is_stable
    
    if verbose:
        if is_stable:
            print("  ✓ Solution is numerically stable")
        else:
            print(f"  ✗ Stability issues found:")
            for issue in issues:
                print(f"    - {issue}")
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        if all_tests_passed:
            print("✓ ALL VALIDATION TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED - See details above")
        print("="*70)
    
    return all_tests_passed, results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Running validation test suite...\n")
    
    # Run full validation with default parameters
    params_dict = params.get_default_params()
    
    all_passed, results = run_full_validation(
        t_span=(0, 100),
        dt=0.5,
        params_dict=params_dict,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("VALIDATION MODULE TEST COMPLETE")
    print("="*70)