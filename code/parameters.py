"""
PARAMETERS.PY
==============
All physical parameters for the permafrost-carbon-climate model.

Values from peer-reviewed literature (see Phase1_ParameterTable.md for sources)
Course: EH605 - Modelling of Earth System & Sustainability
"""

import numpy as np

# ============================================================================
# COMPLETE PARAMETER DICTIONARY
# ============================================================================

params = {
    # === INITIAL CONDITIONS (Carbon Stocks) ===
    'C_atm_0': 600.0,          # Pg C - Pre-industrial atmospheric carbon (280 ppm)
    'C_active_0': 350.0,       # Pg C - Active layer permafrost (0-3m depth)
    'C_deep_0': 1200.0,        # Pg C - Deep permafrost (>3m depth)
    'T_s_0': 268.0,            # K - Initial Arctic temperature (-5°C)
    
    # === DECOMPOSITION KINETICS ===
    'k0_decomp': 0.02,         # 1/yr - Base decomposition rate at T_ref
    'Q10': 2.5,                # - Temperature sensitivity (Q10 rule)
    'T_ref': 273.0,            # K - Reference temperature (0°C)
    
    # === PERMAFROST THAW ===
    'k_thaw': 0.005,           # 1/yr - Thaw transfer coefficient
    'T_threshold': 273.0,      # K - Thaw activation temperature (0°C)
    'k_smooth': 0.5,           # - Smoothing parameter for sigmoid (1/K)
    
    # === OCEAN CARBON UPTAKE ===
    'k_ocean': 0.015,          # 1/yr - Ocean uptake coefficient
    'C_eq': 600.0,             # Pg C - Ocean equilibrium carbon
    
    # === ENERGY BALANCE (Budyko - Adjusted for Arctic) ===
    'S': 1368.0,               # W/m² - Solar constant
    'A': -150.0,               # W/m² - OLR intercept (adjusted for Arctic)
    'B': 1.0,                  # W/(m²·K) - OLR slope
    'C_heat': 5.0e8,           # J/(m²·K) - Heat capacity (larger for stability)
    
    # === ALBEDO (Ice-Albedo Feedback) ===
    'alpha_ice': 0.70,         # - Ice/snow albedo (high reflectivity)
    'alpha_tundra': 0.20,      # - Tundra/land albedo (low reflectivity)
    'T_freeze': 273.0,         # K - Freezing point
    'T_melt': 283.0,           # K - Complete melt temperature
    
    # === CO2 RADIATIVE FORCING ===
    'alpha_CO2': 5.35,         # W/m² - CO2 radiative efficiency (per doubling)
    'C_ref': 600.0,            # Pg C - Reference CO2 (pre-industrial)
    
    # === CONVERSION FACTORS ===
    'seconds_per_year': 3.15e7,      # s/yr
}

# Calculate W/m² to K/yr conversion factor
params['W_per_m2_to_K_per_yr'] = params['seconds_per_year'] / params['C_heat']

# ============================================================================
# EMISSION SCENARIOS
# ============================================================================

def emissions_scenario(t, scenario='RCP4.5'):
    """
    Return anthropogenic emissions at time t [years]
    
    Parameters:
    -----------
    t : float or array
        Time in years (0 = pre-industrial, e.g., 1850)
    scenario : str
        Emission scenario name
        
    Returns:
    --------
    E : float or array
        Emissions rate [Pg C/yr]
    """
    if scenario == 'none':
        return 0.0
    elif scenario == 'baseline':
        return 4.8 * np.exp(0.024 * t)
    elif scenario == 'RCP2.6':
        return 10.0 * np.exp(-0.02 * t)
    elif scenario == 'RCP4.5':
        return 12.0
    elif scenario == 'RCP8.5':
        return 10.0 * (1.0 + 0.02 * t)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

if __name__ == "__main__":
    print("Parameters loaded successfully!")
    print(f"Initial conditions: C_atm={params['C_atm_0']} Pg C, T={params['T_s_0']-273:.1f}°C")