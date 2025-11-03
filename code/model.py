"""
ODE system for Arctic Permafrost-Carbon-Climate model
"""

import numpy as np

def decomposition_rate(T, params):
    """
    Temperature-dependent decomposition rate using Q10 formulation
    
    Parameters:
    T : float - Temperature [K]
    params : dict - Model parameters
    
    Returns:
    k : float - Decomposition rate [1/year]
    """
    k_base = params['k_base']
    Q10 = params['Q10']
    T_ref = params['T_ref']
    
    k = k_base * Q10**((T - T_ref) / 10.0)
    return k


def thaw_flux_factor(T, params):
    """
    Smooth sigmoid function for permafrost thaw (0 to 1)
    
    Parameters:
    T : float - Temperature [K]
    params : dict - Model parameters
    
    Returns:
    factor : float - Thaw flux factor [dimensionless]
    """
    T_threshold = params['T_threshold']
    beta = params['beta_thaw']
    
    factor = 1.0 / (1.0 + np.exp(-beta * (T - T_threshold)))
    return factor


def albedo(T, params):
    """
    Temperature-dependent albedo with smooth transition
    
    Parameters:
    T : float - Temperature [K]
    params : dict - Model parameters
    
    Returns:
    alpha : float - Albedo [dimensionless, 0-1]
    """
    alpha_min = params['alpha_min']
    alpha_max = params['alpha_max']
    T_melt = params['T_melt']
    gamma = params['gamma_albedo']
    
    alpha = alpha_min + (alpha_max - alpha_min) / (1.0 + np.exp(gamma * (T - T_melt)))
    return alpha


def co2_forcing(C_atm, params):
    """
    Radiative forcing from atmospheric CO2
    
    Parameters:
    C_atm : float - Atmospheric carbon [Pg C]
    params : dict - Model parameters
    
    Returns:
    F : float - Radiative forcing [W/m²]
    """
    alpha_co2 = params['alpha_co2']
    C_ref = params['C_ref']
    
    # Avoid log of zero or negative
    if C_atm <= 0:
        C_atm = 1e-10
    
    F = alpha_co2 * np.log(C_atm / C_ref)
    return F


def permafrost_model(state, t, params, E_anthro):
    """
    Right-hand side of coupled ODEs
    
    Parameters:
    state : array [C_atm, C_active, C_deep, T]
    t : float - Time [years]
    params : dict - Model parameters
    E_anthro : float - Anthropogenic emissions [Pg C/year]
    
    Returns:
    derivatives : array [dC_atm/dt, dC_active/dt, dC_deep/dt, dT/dt]
    """
    # Unpack state
    C_atm, C_active, C_deep, T = state
    
    # Ensure physical constraints
    C_atm = max(C_atm, 0)
    C_active = max(C_active, 0)
    C_deep = max(C_deep, 0)
    T = max(T, 200)  # Reasonable lower bound
    
    # Calculate fluxes
    k_decomp = decomposition_rate(T, params)
    thaw_factor = thaw_flux_factor(T, params)
    
    # Decomposition flux: Active layer → Atmosphere (POSITIVE FEEDBACK)
    F_decomp = k_decomp * C_active
    
    # Thaw flux: Deep permafrost → Active layer (POSITIVE FEEDBACK)
    F_thaw = params['k_thaw'] * C_deep * thaw_factor
    
    # Ocean uptake: Atmosphere → Ocean (NEGATIVE FEEDBACK)
    F_ocean = params['k_ocean'] * (C_atm - params['C_ocean_eq'])
    
    # Anthropogenic emissions (FORCING)
    F_anthro = E_anthro
    
    # ODEs for carbon cycle
    dC_atm_dt = F_decomp + F_anthro - F_ocean
    dC_active_dt = F_thaw - F_decomp
    dC_deep_dt = -F_thaw
    
    # Energy balance for temperature
    alpha_surf = albedo(T, params)
    F_co2 = co2_forcing(C_atm, params)
    
    # Incoming solar (absorbed)
    Q_in = params['S'] * (1.0 - alpha_surf)
    
    # Outgoing longwave radiation (NEGATIVE FEEDBACK - Planck)
    Q_out = params['A'] + params['B'] * (T - params['T_budyko'])
    
    # Net energy balance
    Q_net = Q_in - Q_out + F_co2
    
    # Temperature change
    dT_dt = (params['sec_per_year'] / params['C_heat']) * Q_net
    
    return np.array([dC_atm_dt, dC_active_dt, dC_deep_dt, dT_dt])


def jacobian_numerical(state, params, E_anthro, epsilon=1e-6):
    """
    Numerical Jacobian matrix for stability analysis
    
    Parameters:
    state : array - State at equilibrium
    params : dict - Model parameters
    E_anthro : float - Emissions
    epsilon : float - Perturbation size
    
    Returns:
    J : 4x4 array - Jacobian matrix
    """
    n = len(state)
    J = np.zeros((n, n))
    
    f0 = permafrost_model(state, 0, params, E_anthro)
    
    for i in range(n):
        state_perturbed = state.copy()
        state_perturbed[i] += epsilon
        f_perturbed = permafrost_model(state_perturbed, 0, params, E_anthro)
        J[:, i] = (f_perturbed - f0) / epsilon
    
    return J