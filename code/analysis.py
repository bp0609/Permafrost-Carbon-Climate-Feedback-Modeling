"""
analysis.py
Arctic Permafrost-Carbon-Climate Feedback Model
Stability analysis, equilibrium finding, and bifurcation analysis
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from model import permafrost_model
from parameters import PARAMS, SCENARIOS


def find_equilibria(params, forcing_func, use_albedo_feedback=False,
                   initial_guesses=None):
    """
    Find equilibrium points where all derivatives are zero
    
    Parameters:
        params : dict
            Model parameters
        forcing_func : function
            Anthropogenic forcing function (fixed at some time)
        use_albedo_feedback : bool
            Ice-albedo feedback switch
        initial_guesses : list of arrays, optional
            Initial guesses for equilibria
    
    Returns:
        equilibria : list of arrays
            List of equilibrium states found
    """
    
    # Define function to find roots of
    def residual(state):
        """Return derivatives (should be zero at equilibrium)"""
        # For equilibrium, use steady forcing (e.g., t=100)
        derivs = permafrost_model(state, 100, params, forcing_func, 
                                 use_albedo_feedback)
        return derivs
    
    # Default initial guesses if not provided
    if initial_guesses is None:
        initial_guesses = [
            np.array([600, 170, 750, -3]),   # Cool equilibrium
            np.array([700, 150, 600, 0]),    # Intermediate
            np.array([900, 100, 400, 5]),    # Warm equilibrium
        ]
    
    equilibria = []
    
    for guess in initial_guesses:
        try:
            # Solve for equilibrium
            eq = fsolve(residual, guess, full_output=True)
            solution = eq[0]
            info = eq[1]
            
            # Check if solution converged
            if info['fvec'].max() < 1e-6:  # Residual small enough
                # Check if this is a new equilibrium (not duplicate)
                is_new = True
                for existing_eq in equilibria:
                    if np.allclose(solution, existing_eq, atol=1e-3):
                        is_new = False
                        break
                
                if is_new and all(solution >= 0):  # Physical constraint
                    equilibria.append(solution)
        
        except:
            pass  # Failed to converge, skip this guess
    
    return equilibria


def jacobian_matrix(equilibrium, params, forcing_func, 
                    use_albedo_feedback=False, delta=1e-6):
    """
    Calculate Jacobian matrix at equilibrium point using finite differences
    
    J[i,j] = ∂f_i/∂x_j
    
    Parameters:
        equilibrium : array
            Equilibrium state
        params : dict
            Model parameters
        forcing_func : function
            Forcing function
        use_albedo_feedback : bool
            Feedback switch
        delta : float
            Finite difference step size
    
    Returns:
        J : numpy array, shape (4, 4)
            Jacobian matrix
    """
    
    n = len(equilibrium)
    J = np.zeros((n, n))
    
    # Base derivatives at equilibrium
    f0 = permafrost_model(equilibrium, 100, params, forcing_func, 
                         use_albedo_feedback)
    
    # Compute each column of Jacobian
    for j in range(n):
        # Perturb j-th variable
        state_plus = equilibrium.copy()
        state_plus[j] += delta
        
        # Compute perturbed derivatives
        f_plus = permafrost_model(state_plus, 100, params, forcing_func, 
                                 use_albedo_feedback)
        
        # Finite difference approximation
        J[:, j] = (f_plus - f0) / delta
    
    return J


def stability_classification(eigenvalues):
    """
    Classify stability based on eigenvalues
    
    Parameters:
        eigenvalues : array
            Eigenvalues of Jacobian matrix
    
    Returns:
        classification : str
            Stability type
        is_stable : bool
            Whether equilibrium is stable
    """
    
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Check for complex eigenvalues
    has_complex = np.any(np.abs(imag_parts) > 1e-10)
    
    # Check stability (all real parts negative = stable)
    is_stable = np.all(real_parts < 0)
    
    # Classify type
    if is_stable:
        if has_complex:
            classification = "Stable Spiral"
        else:
            classification = "Stable Node"
    else:
        if np.all(real_parts > 0):
            if has_complex:
                classification = "Unstable Spiral"
            else:
                classification = "Unstable Node"
        else:
            classification = "Saddle Point"
    
    return classification, is_stable


def analyze_equilibrium(equilibrium, params, forcing_func, 
                        use_albedo_feedback=False):
    """
    Complete analysis of an equilibrium point
    
    Parameters:
        equilibrium : array
            Equilibrium state
        params : dict
            Parameters
        forcing_func : function
            Forcing
        use_albedo_feedback : bool
            Feedback switch
    
    Returns:
        analysis : dict
            Dictionary with equilibrium properties
    """
    
    # Calculate Jacobian
    J = jacobian_matrix(equilibrium, params, forcing_func, use_albedo_feedback)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(J)
    
    # Classify stability
    classification, is_stable = stability_classification(eigenvalues)
    
    # Calculate characteristic timescales (1/|λ|)
    timescales = 1.0 / np.abs(np.real(eigenvalues))
    
    analysis = {
        'equilibrium': equilibrium,
        'jacobian': J,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'classification': classification,
        'is_stable': is_stable,
        'timescales': timescales
    }
    
    return analysis


def bifurcation_diagram(param_name, param_values, params, 
                       forcing_func_template, use_albedo_feedback=False):
    """
    Create bifurcation diagram by varying a parameter
    
    Parameters:
        param_name : str
            Parameter to vary (e.g., 'Q10', 'k_thaw')
        param_values : array
            Values of parameter to sweep
        params : dict
            Base parameters
        forcing_func_template : function
            Forcing function
        use_albedo_feedback : bool
            Feedback switch
    
    Returns:
        results : dict
            Bifurcation diagram data
    """
    
    equilibria_stable = []
    equilibria_unstable = []
    param_vals_stable = []
    param_vals_unstable = []
    
    print(f"Computing bifurcation diagram for {param_name}...")
    
    for i, p_val in enumerate(param_values):
        # Update parameter
        params_copy = params.copy()
        params_copy[param_name] = p_val
        
        # Find equilibria
        eq_list = find_equilibria(params_copy, forcing_func_template, 
                                  use_albedo_feedback)
        
        # Classify each equilibrium
        for eq in eq_list:
            analysis = analyze_equilibrium(eq, params_copy, forcing_func_template,
                                          use_albedo_feedback)
            
            if analysis['is_stable']:
                equilibria_stable.append(eq)
                param_vals_stable.append(p_val)
            else:
                equilibria_unstable.append(eq)
                param_vals_unstable.append(p_val)
        
        if (i + 1) % max(1, len(param_values)//10) == 0:
            print(f"  Progress: {100*(i+1)//len(param_values)}%")
    
    results = {
        'param_name': param_name,
        'param_values': param_values,
        'equilibria_stable': np.array(equilibria_stable) if equilibria_stable else None,
        'equilibria_unstable': np.array(equilibria_unstable) if equilibria_unstable else None,
        'param_vals_stable': np.array(param_vals_stable) if param_vals_stable else None,
        'param_vals_unstable': np.array(param_vals_unstable) if param_vals_unstable else None,
    }
    
    return results


def plot_bifurcation_diagram(bifurcation_results, state_var_index=3, 
                             save_path=None):
    """
    Plot bifurcation diagram
    
    Parameters:
        bifurcation_results : dict
            Results from bifurcation_diagram()
        state_var_index : int
            Which state variable to plot (0=C_atm, 1=C_active, 2=C_deep, 3=T_s)
        save_path : str, optional
            Path to save figure
    """
    
    var_names = ['C_atm (Pg C)', 'C_active (Pg C)', 'C_deep (Pg C)', 'T_s (°C)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot stable equilibria
    if bifurcation_results['equilibria_stable'] is not None:
        stable_vals = bifurcation_results['equilibria_stable'][:, state_var_index]
        ax.plot(bifurcation_results['param_vals_stable'], stable_vals,
               'b-', linewidth=2, label='Stable', marker='o', markersize=4)
    
    # Plot unstable equilibria
    if bifurcation_results['equilibria_unstable'] is not None:
        unstable_vals = bifurcation_results['equilibria_unstable'][:, state_var_index]
        ax.plot(bifurcation_results['param_vals_unstable'], unstable_vals,
               'r--', linewidth=2, label='Unstable', marker='x', markersize=6)
    
    ax.set_xlabel(bifurcation_results['param_name'], fontsize=12)
    ax.set_ylabel(var_names[state_var_index], fontsize=12)
    ax.set_title(f'Bifurcation Diagram: {var_names[state_var_index]} vs {bifurcation_results["param_name"]}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bifurcation diagram saved to: {save_path}")
    
    return fig


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """Demonstrate stability analysis"""
    
    print("=" * 70)
    print("STABILITY ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    # Use RCP 4.5 forcing
    forcing = SCENARIOS['RCP4.5']
    
    print("\n1. FINDING EQUILIBRIA")
    print("-" * 70)
    
    equilibria = find_equilibria(PARAMS, forcing, use_albedo_feedback=False)
    
    print(f"Found {len(equilibria)} equilibrium point(s):\n")
    
    for i, eq in enumerate(equilibria):
        print(f"Equilibrium {i+1}:")
        print(f"  C_atm:    {eq[0]:.1f} Pg C ({eq[0]/PARAMS['PgC_to_ppm']:.1f} ppm)")
        print(f"  C_active: {eq[1]:.1f} Pg C")
        print(f"  C_deep:   {eq[2]:.1f} Pg C")
        print(f"  T_s:      {eq[3]:.2f} °C")
        print()
    
    if len(equilibria) > 0:
        print("\n2. STABILITY ANALYSIS")
        print("-" * 70)
        
        for i, eq in enumerate(equilibria):
            print(f"\nEquilibrium {i+1} Analysis:")
            
            analysis = analyze_equilibrium(eq, PARAMS, forcing, 
                                          use_albedo_feedback=False)
            
            print(f"  Classification: {analysis['classification']}")
            print(f"  Stable: {analysis['is_stable']}")
            print(f"\n  Eigenvalues:")
            for j, (eig, ts) in enumerate(zip(analysis['eigenvalues'], 
                                              analysis['timescales'])):
                real = np.real(eig)
                imag = np.imag(eig)
                print(f"    λ_{j+1} = {real:+.4f} {imag:+.4f}j  "
                     f"(timescale: {ts:.1f} years)")
        
        print("\n3. CREATING SIMPLE BIFURCATION DIAGRAM")
        print("-" * 70)
        print("Varying Q10 from 1.5 to 4.0...")
        
        # Simple bifurcation: vary Q10
        q10_values = np.linspace(1.5, 4.0, 20)
        
        bifurc_results = bifurcation_diagram(
            'Q10', 
            q10_values, 
            PARAMS,
            forcing,
            use_albedo_feedback=False
        )
        
        # Plot results
        print("\nCreating bifurcation plot...")
        plot_bifurcation_diagram(bifurc_results, state_var_index=3,
                                save_path='../figures/bifurcation_Q10.png')
        
        plt.show()
    
    else:
        print("No equilibria found. System may be non-equilibrium (transient).")
        print("This is expected for time-dependent forcing scenarios.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)