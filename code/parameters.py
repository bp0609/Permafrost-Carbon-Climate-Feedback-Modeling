"""
parameters.py
Arctic Permafrost-Carbon-Climate Feedback Model
All physical constants and model parameters with sources
"""

import numpy as np

# =============================================================================
# INITIAL CONDITIONS (Year 2000)
# =============================================================================

# Carbon stocks [Pg C]
C_ATM_0 = 594.0          # Pre-industrial atmospheric carbon (280 ppm)
C_ACTIVE_0 = 174.0       # Active layer permafrost carbon (Vonk et al. 2024)
C_DEEP_0 = 800.0         # Deep permafrost carbon (Hugelius et al. 2014)

# Temperature [°C]
T_SURFACE_0 = -5.0       # Initial Arctic surface temperature

# =============================================================================
# DECOMPOSITION PARAMETERS
# =============================================================================

# Q10 formulation (recommended for simplicity)
Q10 = 2.6                # Temperature sensitivity coefficient (Bracho et al. 2016)
K_REF = 0.05             # Reference decomposition rate [yr^-1] at T_ref
T_REF = 0.0              # Reference temperature [°C]

# Arrhenius formulation (alternative, more physically grounded)
E_A = 67000.0            # Activation energy [J/mol] (Filimonenko & Kuzyakov 2025)
K_0 = 5.0e13             # Pre-exponential factor [yr^-1] (calibrated)
R_GAS = 8.314            # Gas constant [J/(K·mol)]

# =============================================================================
# CLIMATE PARAMETERS
# =============================================================================

# Solar and radiative parameters
S_0 = 1370.0             # Solar constant [W/m²]
ALBEDO = 0.30            # Mean albedo (no feedback case)

# Budykov coefficients (North et al. 1981)
A_BUDYKOV = 204.0        # Baseline OLR [W/m²]
B_BUDYKOV = 2.17         # Climate feedback parameter [W/(m²·°C)]

# Heat capacity
C_HEAT = 2.08e8          # Effective heat capacity [J/(m²·°C)] (70m mixed layer)

# CO2 radiative forcing (Myhre et al. 1998)
ALPHA_FORCING = 5.35     # CO2 forcing coefficient [W/m²]
C_0 = 594.0              # Pre-industrial reference carbon [Pg C]

# =============================================================================
# ALBEDO FEEDBACK PARAMETERS (Optional)
# =============================================================================

ALPHA_MAX = 0.70         # Ice-covered albedo
ALPHA_MIN = 0.15         # Ice-free albedo
T_ALPHA_CRIT = -2.0      # Transition temperature [°C]
K_ALPHA = 0.3            # Transition sharpness [°C^-1]

# =============================================================================
# PERMAFROST THAW DYNAMICS
# =============================================================================

T_CRIT = 0.0             # Critical thaw temperature [°C]
K_THAW = 0.01            # Thaw rate coefficient [yr^-1]
BETA_THAW = 0.5          # Threshold sharpness [°C^-1]

# =============================================================================
# CARBON CYCLE PARAMETERS
# =============================================================================

K_UPTAKE = 0.02          # Combined ocean+land uptake rate [yr^-1]

# =============================================================================
# CONVERSION FACTORS
# =============================================================================

PGC_TO_PPM = 2.12        # Conversion: 1 ppm CO2 ≈ 2.12 Pg C
SEC_PER_YEAR = 3.156e7   # Seconds per year

# =============================================================================
# ANTHROPOGENIC FORCING SCENARIOS (RCP)
# =============================================================================

def rcp_26_emissions(t):
    """
    RCP 2.6 anthropogenic emissions [Pg C/yr]
    Stringent mitigation scenario
    
    Parameters:
        t : float - Time in years since 2000
    """
    if t < 60:
        return 9.0 - 0.15 * t  # Declining emissions
    elif t < 100:
        return 0.0             # Zero emissions
    else:
        return -2.0            # Net negative emissions


def rcp_45_emissions(t):
    """
    RCP 4.5 anthropogenic emissions [Pg C/yr]
    Moderate mitigation scenario (polynomial approximation)
    
    Parameters:
        t : float - Time in years since 2000
    """
    return 9.5 - 0.08 * t + 0.0003 * t**2


def rcp_85_emissions(t):
    """
    RCP 8.5 anthropogenic emissions [Pg C/yr]
    High emissions scenario (exponential growth)
    
    Parameters:
        t : float - Time in years since 2000
    """
    return 9.0 * np.exp(0.015 * t)


# =============================================================================
# PARAMETER DICTIONARY FOR EASY ACCESS
# =============================================================================

PARAMS = {
    # Initial conditions
    'C_atm_0': C_ATM_0,
    'C_active_0': C_ACTIVE_0,
    'C_deep_0': C_DEEP_0,
    'T_s_0': T_SURFACE_0,
    
    # Decomposition
    'Q10': Q10,
    'k_ref': K_REF,
    'T_ref': T_REF,
    'E_a': E_A,
    'k_0': K_0,
    'R': R_GAS,
    
    # Climate
    'S_0': S_0,
    'albedo': ALBEDO,
    'A': A_BUDYKOV,
    'B': B_BUDYKOV,
    'C_heat': C_HEAT,
    'alpha_F': ALPHA_FORCING,
    'C_0': C_0,
    
    # Albedo feedback
    'alpha_max': ALPHA_MAX,
    'alpha_min': ALPHA_MIN,
    'T_alpha_crit': T_ALPHA_CRIT,
    'k_alpha': K_ALPHA,
    
    # Thaw
    'T_crit': T_CRIT,
    'k_thaw': K_THAW,
    'beta_thaw': BETA_THAW,
    
    # Carbon cycle
    'k_uptake': K_UPTAKE,
    
    # Conversions
    'PgC_to_ppm': PGC_TO_PPM,

    'alpha_0': ALPHA_0,
}

# =============================================================================
# SCENARIO SELECTION
# =============================================================================

SCENARIOS = {
    'RCP2.6': rcp_26_emissions,
    'RCP4.5': rcp_45_emissions,
    'RCP8.5': rcp_85_emissions,
}


if __name__ == "__main__":
    """Test parameter values and print summary"""
    print("=" * 70)
    print("ARCTIC PERMAFROST-CARBON-CLIMATE MODEL PARAMETERS")
    print("=" * 70)
    print("\nINITIAL CONDITIONS:")
    print(f"  Atmospheric C:     {C_ATM_0:.1f} Pg C ({C_ATM_0/PGC_TO_PPM:.1f} ppm)")
    print(f"  Active layer C:    {C_ACTIVE_0:.1f} Pg C")
    print(f"  Deep permafrost C: {C_DEEP_0:.1f} Pg C")
    print(f"  Surface temp:      {T_SURFACE_0:.1f} °C")
    
    print("\nKEY PARAMETERS:")
    print(f"  Q10:               {Q10:.2f}")
    print(f"  k_ref:             {K_REF:.3f} yr^-1")
    print(f"  CO2 forcing coef:  {ALPHA_FORCING:.2f} W/m²")
    print(f"  Thaw threshold:    {T_CRIT:.1f} °C")
    
    print("\nTEST EMISSIONS (year 50 = 2050):")
    for scenario, func in SCENARIOS.items():
        print(f"  {scenario}: {func(50):.2f} Pg C/yr")
    
    print("\n" + "=" * 70)