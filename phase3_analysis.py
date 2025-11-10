"""
Phase 3: Phase Space Analysis & Stability

This module provides tools for:
1. Nullcline calculation
2. Equilibrium point finding
3. Jacobian matrix computation
4. Stability analysis (eigenvalues)
5. Phase portraits with nullclines
"""

import numpy as np
from scipy.optimize import fsolve, root
from scipy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import model
import parameters as params


# ============================================================================
# NULLCLINE CALCULATIONS
# ============================================================================

def calculate_nullclines(params_dict, var_ranges, fixed_vars, t=50):
    """
    Calculate nullclines for phase space analysis
    
    A nullcline is where one derivative equals zero.
    For a 2D phase portrait, we calculate where dX/dt = 0 and dY/dt = 0
    
    Parameters:
    -----------
    params_dict : dict
        Parameter dictionary
    var_ranges : dict
        Dictionary specifying ranges for variables
        Example: {'C_atm': (500, 800), 'T_s': (-5, 15)}
    fixed_vars : dict
        Dictionary specifying fixed values for other variables
        Example: {'C_frozen': 500, 'C_active': 100}
    t : float
        Time point for anthropogenic emissions
    
    Returns:
    --------
    nullclines : dict
        Dictionary containing nullcline data
    """
    # Get variable names and ranges
    var_names = list(var_ranges.keys())
    if len(var_names) != 2:
        raise ValueError("Must specify exactly 2 variables for 2D nullclines")
    
    var1_name, var2_name = var_names
    var1_range, var2_range = var_ranges[var1_name], var_ranges[var2_name]
    
    # Create grid
    n_points = 50
    var1_vals = np.linspace(var1_range[0], var1_range[1], n_points)
    var2_vals = np.linspace(var2_range[0], var2_range[1], n_points)
    Var1, Var2 = np.meshgrid(var1_vals, var2_vals)
    
    # Map variable names to indices
    var_map = {'C_frozen': 0, 'C_active': 1, 'C_atm': 2, 'T_s': 3}
    var1_idx = var_map[var1_name]
    var2_idx = var_map[var2_name]
    
    # Initialize derivative arrays
    dVar1_dt = np.zeros_like(Var1)
    dVar2_dt = np.zeros_like(Var2)
    
    # Calculate derivatives at each grid point
    for i in range(n_points):
        for j in range(n_points):
            # Build state vector
            y = np.zeros(4)
            y[var1_idx] = Var1[i, j]
            y[var2_idx] = Var2[i, j]
            
            # Fill in fixed variables
            for var_name, var_value in fixed_vars.items():
                y[var_map[var_name]] = var_value
            
            # Calculate derivatives
            dydt = model.permafrost_model(y, t, params_dict)
            
            dVar1_dt[i, j] = dydt[var1_idx]
            dVar2_dt[i, j] = dydt[var2_idx]
    
    nullclines = {
        'var1_name': var1_name,
        'var2_name': var2_name,
        'var1_vals': var1_vals,
        'var2_vals': var2_vals,
        'Var1': Var1,
        'Var2': Var2,
        'dVar1_dt': dVar1_dt,
        'dVar2_dt': dVar2_dt,
        'var1_nullcline': (dVar1_dt, 0),  # Where dVar1/dt = 0
        'var2_nullcline': (dVar2_dt, 0),  # Where dVar2/dt = 0
    }
    
    return nullclines


def find_nullcline_curves(nullclines, levels=[0]):
    """
    Extract nullcline curves from nullcline data
    
    Parameters:
    -----------
    nullclines : dict
        Nullcline data from calculate_nullclines
    levels : list
        Contour levels (default [0] for nullclines)
    
    Returns:
    --------
    curves : dict
        Dictionary with 'var1' and 'var2' nullcline curves
    """
    import matplotlib.pyplot as plt
    
    # Create temporary figure for contour extraction
    fig, ax = plt.subplots()
    
    # Extract var1 nullcline
    cs1 = ax.contour(nullclines['Var1'], nullclines['Var2'], 
                     nullclines['dVar1_dt'], levels=levels)
    var1_curves = []
    for collection in cs1.collections:
        for path in collection.get_paths():
            var1_curves.append(path.vertices)
    
    # Extract var2 nullcline
    cs2 = ax.contour(nullclines['Var1'], nullclines['Var2'], 
                     nullclines['dVar2_dt'], levels=levels)
    var2_curves = []
    for collection in cs2.collections:
        for path in collection.get_paths():
            var2_curves.append(path.vertices)
    
    plt.close(fig)
    
    curves = {
        'var1_nullclines': var1_curves,
        'var2_nullclines': var2_curves
    }
    
    return curves


# ============================================================================
# EQUILIBRIUM POINT FINDING
# ============================================================================

def find_equilibrium_numerical(params_dict, initial_guess=None, t=50):
    """
    Find equilibrium point numerically by solving for dY/dt = 0
    
    At equilibrium, all derivatives equal zero:
    dC_frozen/dt = 0
    dC_active/dt = 0
    dC_atm/dt = 0
    dT_s/dt = 0
    
    Parameters:
    -----------
    params_dict : dict
        Parameter dictionary
    initial_guess : array, optional
        Initial guess [C_frozen, C_active, C_atm, T_s]
    t : float
        Time point (for emissions calculation)
    
    Returns:
    --------
    equilibrium : array
        Equilibrium state [C_frozen, C_active, C_atm, T_s]
    success : bool
        Whether convergence was successful
    """
    if initial_guess is None:
        # Use sensible defaults based on parameters
        initial_guess = [
            params_dict['C_frozen_init'] / 2,  # Some thawed
            params_dict['C_active_init'] * 2,   # Some accumulated
            params_dict['CO2_preind_PgC'] * 1.2, # Slightly elevated
            5.0  # Moderate warming
        ]
    
    def equations(y):
        """System of equations to solve: dY/dt = 0"""
        dydt = model.permafrost_model(y, t, params_dict)
        return dydt
    
    # Try multiple methods
    # Method 1: fsolve
    try:
        sol = fsolve(equations, initial_guess, full_output=True)
        equilibrium = sol[0]
        info = sol[1]
        success = (info['fvec']**2).sum() < 1e-6
        
        if success:
            return equilibrium, True
    except:
        pass
    
    # Method 2: root with different algorithms
    methods = ['hybr', 'lm', 'broyden1']
    for method in methods:
        try:
            sol = root(equations, initial_guess, method=method)
            if sol.success:
                return sol.x, True
        except:
            continue
    
    # If all fail, return best attempt
    return initial_guess, False


def find_multiple_equilibria(params_dict, n_guesses=20, t=50):
    """
    Search for multiple equilibrium points with different initial guesses
    
    Parameters:
    -----------
    params_dict : dict
        Parameter dictionary
    n_guesses : int
        Number of random initial guesses to try
    t : float
        Time point
    
    Returns:
    --------
    equilibria : list
        List of found equilibrium points
    """
    equilibria = []
    
    # Try systematic guesses
    C_frozen_vals = np.linspace(0, 1000, 5)
    T_vals = np.linspace(-10, 20, 4)
    
    for C_f in C_frozen_vals:
        for T in T_vals:
            guess = [C_f, 50, 650, T]
            eq, success = find_equilibrium_numerical(params_dict, guess, t)
            
            if success:
                # Check if this is a new equilibrium (not duplicate)
                is_new = True
                for existing_eq in equilibria:
                    if np.allclose(eq, existing_eq, atol=10):
                        is_new = False
                        break
                
                if is_new:
                    equilibria.append(eq)
    
    return equilibria


# ============================================================================
# JACOBIAN MATRIX AND STABILITY ANALYSIS
# ============================================================================

def calculate_jacobian(y, params_dict, t=50, epsilon=1e-6):
    """
    Calculate Jacobian matrix at a point using finite differences
    
    The Jacobian J[i,j] = ∂(dyi/dt)/∂yj
    
    Parameters:
    -----------
    y : array
        State vector [C_frozen, C_active, C_atm, T_s]
    params_dict : dict
        Parameter dictionary
    t : float
        Time point
    epsilon : float
        Step size for finite differences
    
    Returns:
    --------
    J : array
        4x4 Jacobian matrix
    """
    n = len(y)
    J = np.zeros((n, n))
    
    # Calculate base derivatives
    f0 = np.array(model.permafrost_model(y, t, params_dict))
    
    # Calculate partial derivatives
    for j in range(n):
        # Perturb j-th variable
        y_plus = y.copy()
        y_plus[j] += epsilon
        f_plus = np.array(model.permafrost_model(y_plus, t, params_dict))
        
        # Finite difference
        J[:, j] = (f_plus - f0) / epsilon
    
    return J


def analyze_stability(equilibrium, params_dict, t=50):
    """
    Analyze stability of an equilibrium point
    
    Stability determined by eigenvalues of Jacobian:
    - All real parts negative: Stable (attracting)
    - Any real part positive: Unstable (repelling)
    - Purely imaginary: Center (neutrally stable)
    
    Parameters:
    -----------
    equilibrium : array
        Equilibrium point
    params_dict : dict
        Parameter dictionary
    t : float
        Time point
    
    Returns:
    --------
    analysis : dict
        Dictionary containing:
        - jacobian: Jacobian matrix
        - eigenvalues: Complex eigenvalues
        - eigenvectors: Eigenvectors
        - stability: Classification string
        - trace: Trace of Jacobian
        - determinant: Determinant of Jacobian
    """
    # Calculate Jacobian
    J = calculate_jacobian(equilibrium, params_dict, t)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(J)
    
    # Classify stability
    real_parts = np.real(eigenvalues)
    
    if np.all(real_parts < 0):
        stability = "Stable (Sink/Attractor)"
    elif np.all(real_parts > 0):
        stability = "Unstable (Source/Repeller)"
    elif np.any(real_parts > 0):
        stability = "Saddle Point (Unstable)"
    else:
        stability = "Neutrally Stable (Center)"
    
    # Additional properties
    trace = np.trace(J)
    determinant = np.linalg.det(J)
    
    analysis = {
        'jacobian': J,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'stability': stability,
        'trace': trace,
        'determinant': determinant,
        'real_parts': real_parts,
        'imag_parts': np.imag(eigenvalues)
    }
    
    return analysis


# ============================================================================
# PHASE PORTRAITS WITH NULLCLINES
# ============================================================================

def plot_phase_portrait_with_nullclines(var_ranges, fixed_vars, params_dict,
                                       trajectories=None, equilibria=None,
                                       title="Phase Portrait with Nullclines",
                                       filename=None):
    """
    Create phase portrait with nullclines, vector field, and trajectories
    
    Parameters:
    -----------
    var_ranges : dict
        Ranges for the two phase space variables
    fixed_vars : dict
        Fixed values for other variables
    params_dict : dict
        Parameter dictionary
    trajectories : list of tuples, optional
        List of (t, solution) tuples to plot
    equilibria : list of arrays, optional
        List of equilibrium points to mark
    title : str
        Plot title
    filename : str, optional
        Save filename
    """
    import visualization as viz
    viz.set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate nullclines
    print(f"Calculating nullclines for {list(var_ranges.keys())}...")
    nullclines = calculate_nullclines(params_dict, var_ranges, fixed_vars)
    
    var1_name = nullclines['var1_name']
    var2_name = nullclines['var2_name']
    
    # Plot nullclines as contours
    ax.contour(nullclines['Var1'], nullclines['Var2'], 
              nullclines['dVar1_dt'], levels=[0], 
              colors='blue', linewidths=2.5, linestyles='--',
              label=f'd{var1_name}/dt = 0')
    
    ax.contour(nullclines['Var1'], nullclines['Var2'], 
              nullclines['dVar2_dt'], levels=[0], 
              colors='red', linewidths=2.5, linestyles='--',
              label=f'd{var2_name}/dt = 0')
    
    # Plot vector field (direction field)
    n_arrows = 15
    var1_arrow = nullclines['var1_vals'][::len(nullclines['var1_vals'])//n_arrows]
    var2_arrow = nullclines['var2_vals'][::len(nullclines['var2_vals'])//n_arrows]
    
    for v1 in var1_arrow:
        for v2 in var2_arrow:
            # Build state vector
            y = np.zeros(4)
            var_map = {'C_frozen': 0, 'C_active': 1, 'C_atm': 2, 'T_s': 3}
            y[var_map[var1_name]] = v1
            y[var_map[var2_name]] = v2
            
            for var_name, var_value in fixed_vars.items():
                y[var_map[var_name]] = var_value
            
            # Calculate derivative
            dydt = model.permafrost_model(y, 50, params_dict)
            dv1 = dydt[var_map[var1_name]]
            dv2 = dydt[var_map[var2_name]]
            
            # Normalize for visualization
            magnitude = np.sqrt(dv1**2 + dv2**2)
            if magnitude > 1e-10:
                dv1_norm = dv1 / magnitude
                dv2_norm = dv2 / magnitude
                
                # Scale arrow
                scale = min(var_ranges[var1_name][1] - var_ranges[var1_name][0],
                           var_ranges[var2_name][1] - var_ranges[var2_name][0]) * 0.03
                
                ax.arrow(v1, v2, dv1_norm * scale, dv2_norm * scale,
                        head_width=scale*0.5, head_length=scale*0.7,
                        fc='gray', ec='gray', alpha=0.4, linewidth=0.5)
    
    # Plot trajectories if provided
    if trajectories:
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        for idx, (t, sol) in enumerate(trajectories):
            var1_data = sol[:, var_map[var1_name]]
            var2_data = sol[:, var_map[var2_name]]
            
            color = colors[idx % len(colors)]
            ax.plot(var1_data, var2_data, '-', color=color, 
                   linewidth=2, alpha=0.7, label=f'Trajectory {idx+1}')
            
            # Mark start and end
            ax.plot(var1_data[0], var2_data[0], 'o', color=color, 
                   markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            ax.plot(var1_data[-1], var2_data[-1], 's', color=color, 
                   markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot equilibria if provided
    if equilibria:
        for eq in equilibria:
            v1_eq = eq[var_map[var1_name]]
            v2_eq = eq[var_map[var2_name]]
            ax.plot(v1_eq, v2_eq, '*', color='red', markersize=20, 
                   markeredgecolor='black', markeredgewidth=2, 
                   label='Equilibrium', zorder=10)
    
    # Labels and formatting
    var_labels = {
        'C_frozen': 'Frozen Carbon [PgC]',
        'C_active': 'Active Carbon [PgC]',
        'C_atm': 'Atmospheric CO₂ [PgC]',
        'T_s': 'Temperature [°C]'
    }
    
    ax.set_xlabel(var_labels[var1_name], fontsize=13)
    ax.set_ylabel(var_labels[var2_name], fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig, ax


# ============================================================================
# SENSITIVITY TO INITIAL CONDITIONS
# ============================================================================

def test_initial_condition_sensitivity(params_dict, base_ic, perturbations,
                                      t_span=(0, 100), dt=0.5):
    """
    Test sensitivity to initial conditions
    
    Parameters:
    -----------
    params_dict : dict
        Parameter dictionary
    base_ic : array
        Base initial condition
    perturbations : list of arrays
        List of perturbation vectors to add to base_ic
    t_span : tuple
        Time span
    dt : float
        Time step
    
    Returns:
    --------
    results : list
        List of (t, solution) tuples for each initial condition
    """
    results = []
    
    # Base case
    print(f"Running base case...")
    t, sol = model.run_simulation(t_span, y0=base_ic, params_dict=params_dict, dt=dt)
    results.append((t, sol))
    
    # Perturbed cases
    for idx, pert in enumerate(perturbations):
        print(f"Running perturbation {idx+1}/{len(perturbations)}...")
        ic = base_ic + pert
        # Ensure non-negative
        ic = np.maximum(ic, 0)
        t, sol = model.run_simulation(t_span, y0=ic, params_dict=params_dict, dt=dt)
        results.append((t, sol))
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_equilibrium_analysis(equilibrium, analysis):
    """
    Print detailed analysis of an equilibrium point
    
    Parameters:
    -----------
    equilibrium : array
        Equilibrium point
    analysis : dict
        Analysis dictionary from analyze_stability
    """
    print("="*70)
    print("EQUILIBRIUM POINT ANALYSIS")
    print("="*70)
    
    print("\nEquilibrium State:")
    print(f"  C_frozen = {equilibrium[0]:.2f} PgC")
    print(f"  C_active = {equilibrium[1]:.2f} PgC")
    print(f"  C_atm = {equilibrium[2]:.2f} PgC")
    print(f"  T_s = {equilibrium[3]:.2f} °C")
    
    print("\nStability Classification:")
    print(f"  {analysis['stability']}")
    
    print("\nEigenvalues:")
    for idx, (eval_val, real_part) in enumerate(zip(analysis['eigenvalues'], 
                                                     analysis['real_parts'])):
        sign = "Stable" if real_part < 0 else "Unstable"
        print(f"  λ{idx+1} = {eval_val.real:.4f} + {eval_val.imag:.4f}i  ({sign})")
    
    print("\nJacobian Properties:")
    print(f"  Trace = {analysis['trace']:.4f}")
    print(f"  Determinant = {analysis['determinant']:.4f}")
    
    print("\nJacobian Matrix:")
    print("     dC_f    dC_a    dC_atm   dT_s")
    for i, row_name in enumerate(['dC_f/dt', 'dC_a/dt', 'dC_atm/dt', 'dT_s/dt']):
        print(f"{row_name:10s}", end='')
        for j in range(4):
            print(f"{analysis['jacobian'][i,j]:8.4f}", end='')
        print()
    
    print("="*70)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Phase 3 analysis tools...\n")
    
    # Load parameters
    params_dict = params.get_default_params()
    
    # Test equilibrium finding
    print("="*70)
    print("TEST 1: Finding Equilibrium Points")
    print("="*70)
    
    eq, success = find_equilibrium_numerical(params_dict)
    print(f"\nEquilibrium found: {success}")
    if success:
        print(f"C_frozen = {eq[0]:.2f} PgC")
        print(f"C_active = {eq[1]:.2f} PgC")
        print(f"C_atm = {eq[2]:.2f} PgC")
        print(f"T_s = {eq[3]:.2f} °C")
    
    # Test stability analysis
    print("\n" + "="*70)
    print("TEST 2: Stability Analysis")
    print("="*70)
    
    if success:
        analysis = analyze_stability(eq, params_dict)
        print_equilibrium_analysis(eq, analysis)
    
    print("\n✓ Phase 3 analysis module ready!")