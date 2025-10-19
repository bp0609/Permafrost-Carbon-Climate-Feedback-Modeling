"""
helper_functions.py
Arctic Permafrost-Carbon-Climate Feedback Model
Individual component functions for model physics
"""

import numpy as np


def decomposition_rate_q10(T_s, params):
    """
    Calculate temperature-dependent decomposition rate using Q10 formulation
    
    k(T) = k_ref * Q10^((T - T_ref) / 10)
    
    Parameters:
        T_s : float - Surface temperature [°C]
        params : dict - Parameter dictionary
    
    Returns:
        k : float - Decomposition rate [yr^-1]
    
    Reference: Bracho et al. 2016
    """
    Q10 = params['Q10']
    k_ref = params['k_ref']
    T_ref = params['T_ref']
    
    k = k_ref * (Q10 ** ((T_s - T_ref) / 10.0))
    
    return k


def decomposition_rate_arrhenius(T_s, params):
    """
    Calculate temperature-dependent decomposition rate using Arrhenius equation
    
    k(T) = k_0 * exp(-E_a / (R * T_K))
    
    Parameters:
        T_s : float - Surface temperature [°C]
        params : dict - Parameter dictionary
    
    Returns:
        k : float - Decomposition rate [yr^-1]
    
    Reference: Filimonenko & Kuzyakov 2025
    """
    k_0 = params['k_0']
    E_a = params['E_a']
    R = params['R']
    
    # Convert temperature to Kelvin
    T_K = T_s + 273.15
    
    k = k_0 * np.exp(-E_a / (R * T_K))
    
    return k


def radiative_forcing_co2(C_atm, params):
    """
    Calculate CO2 radiative forcing using logarithmic relationship
    
    ΔF = α_F * ln(C_atm / C_0)
    
    Parameters:
        C_atm : float - Atmospheric carbon [Pg C]
        params : dict - Parameter dictionary
    
    Returns:
        ΔF : float - Radiative forcing [W/m²]
    
    Reference: Myhre et al. 1998 (IPCC standard)
    """
    alpha_F = params['alpha_F']
    C_0 = params['C_0']
    
    # Prevent log of zero or negative values
    if C_atm <= 0:
        return 0.0
    
    delta_F = alpha_F * np.log(C_atm / C_0)
    
    return delta_F


def albedo_constant(T_s, params):
    """
    Return constant albedo (no ice-albedo feedback)
    
    Parameters:
        T_s : float - Surface temperature [°C]
        params : dict - Parameter dictionary
    
    Returns:
        alpha : float - Albedo [dimensionless]
    """
    return params['albedo']


def albedo_feedback(T_s, params):
    """
    Calculate temperature-dependent albedo (ice-albedo feedback)
    
    α(T) = α_min + (α_max - α_min) / (1 + exp(k_α * (T - T_α_crit)))
    
    Parameters:
        T_s : float - Surface temperature [°C]
        params : dict - Parameter dictionary
    
    Returns:
        alpha : float - Albedo [dimensionless, 0-1]
    
    Physical interpretation:
        - Cold temps (T << T_crit): α ≈ α_max (ice-covered)
        - Warm temps (T >> T_crit): α ≈ α_min (ice-free)
    """
    alpha_max = params['alpha_max']
    alpha_min = params['alpha_min']
    T_alpha_crit = params['T_alpha_crit']
    k_alpha = params['k_alpha']
    
    alpha = alpha_min + (alpha_max - alpha_min) / (
        1.0 + np.exp(k_alpha * (T_s - T_alpha_crit))
    )
    
    return alpha


def thaw_flux(T_s, C_deep, params):
    """
    Calculate carbon flux from deep permafrost to active layer
    
    F_thaw = k_thaw * C_deep * f(T)
    where f(T) = 1 / (1 + exp(-β * (T - T_crit)))
    
    Parameters:
        T_s : float - Surface temperature [°C]
        C_deep : float - Deep permafrost carbon [Pg C]
        params : dict - Parameter dictionary
    
    Returns:
        F_thaw : float - Thaw flux [Pg C/yr]
    
    Physical interpretation:
        - T < T_crit: Very little thaw
        - T = T_crit: Half-maximum thaw rate
        - T > T_crit: Significant thaw
    """
    k_thaw = params['k_thaw']
    T_crit = params['T_crit']
    beta = params['beta_thaw']
    
    # Sigmoid function for smooth threshold
    f_T = 1.0 / (1.0 + np.exp(-beta * (T_s - T_crit)))
    
    F_thaw = k_thaw * C_deep * f_T
    
    return F_thaw


def carbon_uptake(C_atm, params):
    """
    Calculate combined ocean and terrestrial carbon uptake
    
    F_uptake = k_uptake * (C_atm - C_0)
    
    Parameters:
        C_atm : float - Atmospheric carbon [Pg C]
        params : dict - Parameter dictionary
    
    Returns:
        F_uptake : float - Uptake flux [Pg C/yr]
    
    Note: Positive values indicate removal from atmosphere
    """
    k_uptake = params['k_uptake']
    C_0 = params['C_0']
    
    F_uptake = k_uptake * (C_atm - C_0)
    
    return F_uptake


def incoming_solar(T_s, params, use_albedo_feedback=False):
    """
    Calculate absorbed solar radiation
    
    S_in = (S_0 / 4) * (1 - α)
    
    Parameters:
        T_s : float - Surface temperature [°C]
        params : dict - Parameter dictionary
        use_albedo_feedback : bool - Whether to include ice-albedo feedback
    
    Returns:
        S_in : float - Absorbed solar radiation [W/m²]
    """
    S_0 = params['S_0']
    
    if use_albedo_feedback:
        alpha = albedo_feedback(T_s, params)
    else:
        alpha = albedo_constant(T_s, params)
    
    S_in = (S_0 / 4.0) * (1.0 - alpha)
    
    return S_in


def outgoing_longwave(T_s, params):
    """
    Calculate outgoing longwave radiation using Budykov linearization
    
    OLR = A + B * T
    
    Parameters:
        T_s : float - Surface temperature [°C]
        params : dict - Parameter dictionary
    
    Returns:
        OLR : float - Outgoing longwave radiation [W/m²]
    
    Reference: North et al. 1981
    """
    A = params['A']
    B = params['B']
    
    OLR = A + B * T_s
    
    return OLR


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def carbon_to_ppm(C_atm, params):
    """Convert atmospheric carbon to CO2 concentration"""
    return C_atm / params['PgC_to_ppm']


def ppm_to_carbon(co2_ppm, params):
    """Convert CO2 concentration to atmospheric carbon"""
    return co2_ppm * params['PgC_to_ppm']


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

if __name__ == "__main__":
    """Test all helper functions with sample values"""
    from parameters import PARAMS
    
    print("=" * 70)
    print("TESTING HELPER FUNCTIONS")
    print("=" * 70)
    
    # Test decomposition rates
    T_test = 5.0  # 5°C
    print(f"\n1. DECOMPOSITION RATE at T = {T_test}°C:")
    k_q10 = decomposition_rate_q10(T_test, PARAMS)
    k_arr = decomposition_rate_arrhenius(T_test, PARAMS)
    print(f"   Q10 formulation:      {k_q10:.4f} yr^-1")
    print(f"   Arrhenius formulation: {k_arr:.4e} yr^-1")
    
    # Test over temperature range
    print("\n2. DECOMPOSITION SENSITIVITY:")
    for T in [-10, -5, 0, 5, 10]:
        k = decomposition_rate_q10(T, PARAMS)
        print(f"   T = {T:3d}°C:  k = {k:.4f} yr^-1")
    
    # Test radiative forcing
    print("\n3. CO2 RADIATIVE FORCING:")
    for C_mult in [1.0, 1.5, 2.0]:
        C = PARAMS['C_0'] * C_mult
        co2_ppm = carbon_to_ppm(C, PARAMS)
        dF = radiative_forcing_co2(C, PARAMS)
        print(f"   {co2_ppm:.0f} ppm: ΔF = {dF:.2f} W/m²")
    
    # Test albedo
    print("\n4. ALBEDO (with feedback):")
    for T in [-10, -5, 0, 5, 10]:
        alpha = albedo_feedback(T, PARAMS)
        print(f"   T = {T:3d}°C:  α = {alpha:.3f}")
    
    # Test thaw flux
    print("\n5. THAW FLUX (C_deep = 800 Pg C):")
    C_deep_test = 800.0
    for T in [-5, -2, 0, 2, 5]:
        F = thaw_flux(T, C_deep_test, PARAMS)
        print(f"   T = {T:3d}°C:  F_thaw = {F:.3f} Pg C/yr")
    
    # Test energy balance components
    print("\n6. ENERGY BALANCE at T = 0°C:")
    T = 0.0
    S_in = incoming_solar(T, PARAMS, use_albedo_feedback=False)
    OLR = outgoing_longwave(T, PARAMS)
    dF_co2 = radiative_forcing_co2(PARAMS['C_0'] * 1.5, PARAMS)
    print(f"   Incoming solar:    {S_in:.2f} W/m²")
    print(f"   Outgoing longwave: {OLR:.2f} W/m²")
    print(f"   CO2 forcing:       {dF_co2:.2f} W/m²")
    print(f"   Net:               {S_in - OLR + dF_co2:.2f} W/m²")
    
    print("\n" + "=" * 70)