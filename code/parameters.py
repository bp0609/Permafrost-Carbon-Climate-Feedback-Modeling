"""
Model parameters for Arctic Permafrost-Carbon-Climate model
All values from peer-reviewed literature
"""

import numpy as np

# ============================================================
# CARBON CYCLE PARAMETERS
# ============================================================

# Decomposition (temperature-dependent)
K_BASE = 0.02          # Base decomposition rate [1/year] at 0°C
Q10 = 2.5              # Temperature sensitivity factor [-]
T_REF = 273.0          # Reference temperature [K] (0°C)

# Permafrost thaw
K_THAW = 0.003         # Thaw rate coefficient [1/year]
T_THRESHOLD = 273.0    # Thaw threshold temperature [K] (0°C)
BETA_THAW = 0.5        # Sigmoid steepness [1/K]

# Ocean uptake
K_OCEAN = 0.015        # Ocean uptake coefficient [1/year]
C_OCEAN_EQ = 600.0     # Ocean equilibrium carbon [Pg C]

# ============================================================
# ENERGY BALANCE PARAMETERS
# ============================================================

# Solar radiation (Arctic-specific, annual average)
SOLAR_EFFECTIVE = 240.0    # Arctic effective solar input [W/m²]

# Albedo
ALPHA_MIN = 0.15           # Minimum albedo (dark tundra) [-]
ALPHA_MAX = 0.65           # Maximum albedo (ice/snow) [-]
T_MELT = 273.0             # Melting point for albedo transition [K]
GAMMA_ALBEDO = 0.2         # Albedo transition steepness [1/K]

# Outgoing longwave (Budyko approximation)
A_BUDYKO = 132.0           # OLR intercept [W/m²]
B_BUDYKO = 2.0             # OLR slope [W/(m²·K)]
T_BUDYKO = 273.0           # Reference temperature for OLR [K]

# Heat capacity
C_HEAT = 5.0e7             # Effective heat capacity [J/(m²·K)]

# CO2 forcing
ALPHA_CO2 = 5.35           # CO2 radiative forcing coefficient [W/m²]
C_REF = 600.0              # Reference atmospheric carbon [Pg C]

# Conversion factor
SECONDS_PER_YEAR = 3.156e7 # [s/year]

# ============================================================
# INITIAL CONDITIONS
# ============================================================

# Pre-industrial
C_ATM_0 = 600.0        # Atmospheric carbon [Pg C] (~280 ppm)
C_ACTIVE_0 = 200.0     # Active layer permafrost [Pg C]
C_DEEP_0 = 1400.0      # Deep permafrost [Pg C]
T_SURFACE_0 = 270.0    # Surface temperature [K] (-3°C)

# ============================================================
# EMISSION SCENARIOS
# ============================================================

# Anthropogenic emissions [Pg C/year]
E_RCP26 = 5.0          # Low emissions
E_BASELINE = 10.0      # Baseline
E_RCP85 = 20.0         # High emissions

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

T_FINAL = 300.0        # Simulation duration [years]
N_POINTS = 3000        # Number of time points

def get_params_dict():
    """Return all parameters as a dictionary"""
    return {
        'k_base': K_BASE,
        'Q10': Q10,
        'T_ref': T_REF,
        'k_thaw': K_THAW,
        'T_threshold': T_THRESHOLD,
        'beta_thaw': BETA_THAW,
        'k_ocean': K_OCEAN,
        'C_ocean_eq': C_OCEAN_EQ,
        'S': SOLAR_EFFECTIVE,
        'alpha_min': ALPHA_MIN,
        'alpha_max': ALPHA_MAX,
        'T_melt': T_MELT,
        'gamma_albedo': GAMMA_ALBEDO,
        'A': A_BUDYKO,
        'B': B_BUDYKO,
        'T_budyko': T_BUDYKO,
        'C_heat': C_HEAT,
        'alpha_co2': ALPHA_CO2,
        'C_ref': C_REF,
        'sec_per_year': SECONDS_PER_YEAR
    }

def get_initial_conditions():
    """Return initial conditions"""
    return np.array([C_ATM_0, C_ACTIVE_0, C_DEEP_0, T_SURFACE_0])