"""
Core Model Implementation: Arctic Permafrost-Carbon-Climate System

4 Coupled ODEs:
1. dC_frozen/dt = -k_thaw(T) * C_frozen
2. dC_active/dt = k_thaw(T) * C_frozen - k_decomp(T) * C_active
3. dC_atm/dt = k_decomp(T) * C_active - C_atm/τ + E_anthro(t)
4. dT/dt = [Q_in - Q_out + ΔF_CO2(C_atm)] / C_heat

Based on:
- Lecture 3: Energy balance
- Lecture 4: Feedback mechanisms
- Lecture 8: Coupled dynamical systems
- Lecture 11: Carbon cycle box models
"""

import numpy as np
from scipy.integrate import odeint
import parameters as params


# ============================================================================
# TEMPERATURE-DEPENDENT RATE FUNCTIONS
# ============================================================================

def thaw_rate(T, params_dict):
    """
    Calculate temperature-dependent permafrost thawing rate
    
    Uses Arrhenius-type exponential temperature dependence:
    k_thaw(T) = k0 * exp(β * (T - T_ref))
    
    Parameters:
    -----------
    T : float
        Temperature [°C]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    k_thaw : float
        Thawing rate [1/year]
    
    Reference:
    - Similar to Lecture 8 temperature-dependent population growth
    - Exponential form common in thermal activation processes
    """
    if not params_dict['enable_permafrost_thaw']:
        # If feedback disabled, use base rate only
        return params_dict['k_thaw_base']
    
    k0 = params_dict['k_thaw_base']
    beta = params_dict['beta_thaw']
    T_ref = params_dict['T_ref']
    
    k_thaw = k0 * np.exp(beta * (T - T_ref))
    
    return k_thaw


def decomposition_rate(T, params_dict):
    """
    Calculate temperature-dependent decomposition rate using Q10 formulation
    
    Q10 approach: rate increases by factor Q10 for every 10°C increase
    k_decomp(T) = k0 * Q10^((T - T_ref)/10)
    
    Parameters:
    -----------
    T : float
        Temperature [°C]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    k_decomp : float
        Decomposition rate [1/year]
    
    Reference:
    - Lecture 11: Biological respiration rates
    - Q10 = 2 is typical for biological processes
    """
    if not params_dict['enable_temp_decomposition']:
        # If feedback disabled, use base rate only
        return params_dict['k_decomp_base']
    
    k0 = params_dict['k_decomp_base']
    Q10 = params_dict['Q10_decomp']
    T_ref = params_dict['T_ref']
    
    k_decomp = k0 * (Q10 ** ((T - T_ref) / 10.0))
    
    return k_decomp


# ============================================================================
# ALBEDO AND RADIATIVE FORCING
# ============================================================================

def albedo(T, params_dict):
    """
    Calculate albedo as a function of temperature (ice-albedo feedback)
    
    Uses smooth transition between ice-free and ice-covered states:
    α(T) = α_min + (α_max - α_min) / (1 + exp((T - T_mid)/ΔT))
    
    Parameters:
    -----------
    T : float
        Temperature [°C]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    alpha : float
        Albedo (dimensionless, 0-1)
    
    Reference:
    - Lecture 4, Slides 18-22: Ice-albedo feedback
    - Sigmoid function for smooth transition
    """
    if not params_dict['enable_albedo_feedback']:
        # If feedback disabled, use average albedo
        return (params_dict['alpha_min'] + params_dict['alpha_max']) / 2.0
    
    alpha_min = params_dict['alpha_min']
    alpha_max = params_dict['alpha_max']
    T_mid = params_dict['T_albedo_mid']
    dT = params_dict['T_albedo_width']
    
    # Sigmoid transition
    alpha = alpha_min + (alpha_max - alpha_min) / (1.0 + np.exp((T - T_mid) / dT))
    
    return alpha


def CO2_forcing(C_atm, params_dict):
    """
    Calculate radiative forcing from atmospheric CO2
    
    ΔF = α_CO2 * ln(C_atm / C_0)
    
    Parameters:
    -----------
    C_atm : float
        Atmospheric CO2 [PgC]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    Delta_F : float
        Radiative forcing [W/m²]
    
    Reference:
    - Lecture 4, Slide 12-14: CO2 radiative forcing
    - Logarithmic relationship is standard in climate science
    """
    if not params_dict['enable_CO2_forcing']:
        # If feedback disabled, no additional forcing
        return 0.0
    
    alpha_CO2 = params_dict['alpha_CO2']
    C_0 = params_dict['CO2_preind_PgC']
    
    # Avoid log of zero or negative
    if C_atm <= 0:
        return 0.0
    
    Delta_F = alpha_CO2 * np.log(C_atm / C_0)
    
    return Delta_F


def anthropogenic_emissions(t, params_dict):
    """
    Calculate anthropogenic CO2 emissions as a function of time
    
    Simple exponential growth model:
    E(t) = E_0 * exp(r * t)
    
    Parameters:
    -----------
    t : float
        Time [years]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    E : float
        Emission rate [PgC/year]
    """
    E_0 = params_dict['E_2020']
    r = params_dict['E_growth_rate']
    
    E = E_0 * np.exp(r * t)
    
    return E


# ============================================================================
# ENERGY BALANCE COMPONENTS
# ============================================================================

def incoming_solar(alpha_val, params_dict):
    """
    Calculate absorbed solar radiation
    
    Q_in = S/4 * (1 - α)
    
    Factor of 1/4 accounts for:
    - Spherical geometry (factor of 1/2)
    - Day-night cycle (factor of 1/2)
    
    Parameters:
    -----------
    alpha_val : float
        Albedo
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    Q_in : float
        Absorbed solar radiation [W/m²]
    
    Reference:
    - Lecture 3, Slide 9: Global energy balance
    """
    S = params_dict['solar_constant']
    Q_in = (S / 4.0) * (1.0 - alpha_val)
    
    return Q_in


def outgoing_longwave(T, params_dict):
    """
    Calculate outgoing longwave radiation
    
    Uses a balanced formulation that represents Earth's energy budget:
    Q_out = Q_ref + λ * (T - T_ref)
    
    where Q_ref is chosen to balance incoming solar at reference temperature
    
    Parameters:
    -----------
    T : float
        Temperature [°C]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    Q_out : float
        Outgoing longwave radiation [W/m²]
    
    Reference:
    - Lecture 3: Energy balance
    - Lecture 4: Climate feedback parameter
    """
    lambda_clim = params_dict['lambda_climate']
    T_ref = params_dict['T_ref']
    
    # Reference outgoing radiation (balanced at T_ref with average albedo)
    # This should equal incoming solar at equilibrium
    alpha_ref = (params_dict['alpha_min'] + params_dict['alpha_max']) / 2.0
    S = params_dict['solar_constant']
    Q_ref = (S / 4.0) * (1.0 - alpha_ref)  # Should be ~239 W/m²
    
    # Linearized form around reference
    Q_out = Q_ref + lambda_clim * (T - T_ref)
    
    return Q_out


# ============================================================================
# MAIN MODEL: 4 COUPLED ODEs
# ============================================================================

def permafrost_model(y, t, params_dict):
    """
    Main model function defining the 4 coupled ODEs
    
    State variables:
    ----------------
    y[0] = C_frozen  : Frozen permafrost carbon [PgC]
    y[1] = C_active  : Active layer carbon [PgC]
    y[2] = C_atm     : Atmospheric CO2 [PgC]
    y[3] = T_s       : Surface temperature [°C]
    
    Parameters:
    -----------
    y : array
        State vector [C_frozen, C_active, C_atm, T_s]
    t : float
        Time [years]
    params_dict : dict
        Parameter dictionary
    
    Returns:
    --------
    dydt : array
        Derivatives [dC_frozen/dt, dC_active/dt, dC_atm/dt, dT_s/dt]
    
    ODEs:
    -----
    1. dC_frozen/dt = -k_thaw(T) * C_frozen
       (Frozen carbon thaws with temperature-dependent rate)
    
    2. dC_active/dt = k_thaw(T) * C_frozen - k_decomp(T) * C_active
       (Active layer gains from thawing, loses from decomposition)
    
    3. dC_atm/dt = k_decomp(T) * C_active - C_atm/τ + E_anthro(t)
       (Atmosphere gains from decomposition and emissions, loses to ocean)
    
    4. dT/dt = [Q_in(α(T)) - Q_out(T) + ΔF_CO2(C_atm)] / C_heat
       (Temperature responds to energy balance)
    """
    # Unpack state variables
    C_frozen, C_active, C_atm, T_s = y
    
    # Ensure non-negative values (physical constraint)
    C_frozen = max(0.0, C_frozen)
    C_active = max(0.0, C_active)
    C_atm = max(0.0, C_atm)
    
    # Calculate temperature-dependent rates
    k_thaw = thaw_rate(T_s, params_dict)
    k_decomp = decomposition_rate(T_s, params_dict)
    
    # Calculate albedo
    alpha_val = albedo(T_s, params_dict)
    
    # Calculate radiative forcing
    Delta_F = CO2_forcing(C_atm, params_dict)
    
    # Calculate anthropogenic emissions
    E_anthro = anthropogenic_emissions(t, params_dict)
    
    # Energy balance components
    Q_in = incoming_solar(alpha_val, params_dict)
    Q_out = outgoing_longwave(T_s, params_dict)
    
    # Get other parameters
    tau_atm_ocean = params_dict['tau_atm_ocean']
    C_heat = params_dict['C_heat']
    
    # ========================================================================
    # DEFINE THE 4 ODEs
    # ========================================================================
    
    # ODE 1: Frozen permafrost carbon
    dC_frozen_dt = -k_thaw * C_frozen
    
    # ODE 2: Active layer carbon
    dC_active_dt = k_thaw * C_frozen - k_decomp * C_active
    
    # ODE 3: Atmospheric CO2
    # Ocean uptake proportional to excess CO2 above pre-industrial
    C_excess = C_atm - params_dict['CO2_preind_PgC']
    ocean_uptake = C_excess / tau_atm_ocean if C_excess > 0 else 0
    dC_atm_dt = k_decomp * C_active - ocean_uptake + E_anthro
    
    # ODE 4: Surface temperature
    dT_dt = (Q_in - Q_out + Delta_F) / C_heat
    
    # Return derivatives
    dydt = [dC_frozen_dt, dC_active_dt, dC_atm_dt, dT_dt]
    
    return dydt


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_simulation(t_span, y0=None, params_dict=None, dt=0.1):
    """
    Run the permafrost-carbon-climate model simulation
    
    Parameters:
    -----------
    t_span : tuple
        (t_start, t_end) time span [years]
    y0 : array, optional
        Initial conditions [C_frozen, C_active, C_atm, T_s]
        If None, uses default from parameters
    params_dict : dict, optional
        Parameter dictionary
        If None, uses default parameters
    dt : float, optional
        Time step [years]
    
    Returns:
    --------
    t : array
        Time points [years]
    y : array
        Solution array, shape (len(t), 4)
        Columns: [C_frozen, C_active, C_atm, T_s]
    """
    # Get default parameters if not provided
    if params_dict is None:
        params_dict = params.get_default_params()
    
    # Set default initial conditions if not provided
    if y0 is None:
        y0 = [
            params_dict['C_frozen_init'],
            params_dict['C_active_init'],
            params_dict['CO2_preind_PgC'],
            params_dict['T_init']
        ]
    
    # Create time array
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    
    # Solve ODEs
    solution = odeint(permafrost_model, y0, t, args=(params_dict,))
    
    return t, solution


def run_multiple_scenarios(scenarios_dict, t_span, dt=0.1):
    """
    Run multiple scenarios with different parameter settings
    
    Parameters:
    -----------
    scenarios_dict : dict
        Dictionary of scenarios, each containing a params_dict
        Example: {'baseline': params1, 'high_emissions': params2}
    t_span : tuple
        (t_start, t_end) time span [years]
    dt : float, optional
        Time step [years]
    
    Returns:
    --------
    results : dict
        Dictionary with same keys as scenarios_dict
        Values are tuples (t, solution)
    """
    results = {}
    
    for scenario_name, params_dict in scenarios_dict.items():
        t, solution = run_simulation(t_span, params_dict=params_dict, dt=dt)
        results[scenario_name] = (t, solution)
    
    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing model components...")
    
    # Load default parameters
    params_dict = params.get_default_params()
    
    # Test temperature-dependent functions
    print("\n--- Testing Temperature-Dependent Functions ---")
    T_test = np.array([-10, -5, 0, 5, 10])
    
    print("Temperature [°C]:", T_test)
    print("Thaw rate [1/yr]:", [f"{thaw_rate(T, params_dict):.4f}" for T in T_test])
    print("Decomp rate [1/yr]:", [f"{decomposition_rate(T, params_dict):.4f}" for T in T_test])
    print("Albedo:", [f"{albedo(T, params_dict):.3f}" for T in T_test])
    
    # Test CO2 forcing
    print("\n--- Testing CO2 Forcing ---")
    C_test = np.array([600, 700, 800, 900, 1000])
    print("C_atm [PgC]:", C_test)
    print("ΔF [W/m²]:", [f"{CO2_forcing(C, params_dict):.2f}" for C in C_test])
    
    # Test short simulation
    print("\n--- Running Short Test Simulation (10 years) ---")
    t, sol = run_simulation((0, 10), dt=0.1)
    
    print(f"Time points: {len(t)}")
    print(f"Initial state: C_frozen={sol[0,0]:.1f}, C_active={sol[0,1]:.1f}, "
          f"C_atm={sol[0,2]:.1f}, T={sol[0,3]:.2f}")
    print(f"Final state: C_frozen={sol[-1,0]:.1f}, C_active={sol[-1,1]:.1f}, "
          f"C_atm={sol[-1,2]:.1f}, T={sol[-1,3]:.2f}")
    
    print("\n✓ Model implementation complete and functional!")