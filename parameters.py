"""
Parameter Definitions for Arctic Permafrost-Carbon-Climate Model

All parameters are backed by course materials (Lectures 3, 4, 8, 11)
and scientific literature.

References:
- Lecture 3: Earth Surface Temperature & Energy Balance
- Lecture 4: Dynamical Systems & Feedback
- Lecture 8: Lotka-Volterra & Population Dynamics (for rate constants)
- Lecture 11: Carbon Cycle 2 and 3 Box Model
- IPCC AR6 (2021) for climate sensitivity parameters
"""

import numpy as np

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Stefan-Boltzmann constant [W/(m²·K⁴)]
SIGMA = 5.67e-8  # Lecture 3, Slide 8

# Solar constant [W/m²]
SOLAR_CONSTANT = 1368.0  # Lecture 3, Slide 9

# Pre-industrial CO2 concentration [ppm and PgC]
CO2_PREINDUSTRIAL_PPM = 280.0  # Lecture 4, Slide 12
CO2_PREINDUSTRIAL_PGC = 600.0  # ~280 ppm, Lecture 11, Slide 8

# Conversion factor: 1 ppm CO2 ≈ 2.124 PgC (atmosphere)
PPM_TO_PGC = 2.124  # Lecture 11

# Reference temperature for rate calculations [°C]
T_REFERENCE = 0.0  # Freezing point

# ============================================================================
# CARBON CYCLE PARAMETERS
# ============================================================================

class CarbonParams:
    """
    Parameters for permafrost carbon dynamics
    
    Based on:
    - Lecture 11: Carbon cycle box models and residence times
    - Schuur et al. (2015): Permafrost carbon stocks
    """
    
    # Initial carbon stocks [PgC]
    C_FROZEN_INIT = 1000.0      # Frozen permafrost carbon pool
                                 # ~1000-1500 PgC in permafrost (literature)
    
    C_ACTIVE_INIT = 50.0        # Active layer carbon (small initial pool)
                                 # This grows as permafrost thaws
    
    # Thawing dynamics
    k_thaw_base = 0.01          # Base thaw rate [1/year] at T_ref
                                 # Represents ~1% per year baseline thaw
    
    beta_thaw = 0.15            # Temperature sensitivity of thawing [1/°C]
                                 # Exponential temperature dependence
                                 # Based on Arrhenius-type relationships
    
    # Decomposition dynamics (Lecture 11, Slide 15 - respiration)
    k_decomp_base = 0.05        # Base decomposition rate [1/year]
                                 # ~5% per year is typical for soil carbon
    
    Q10_decomp = 2.0            # Q10 factor for decomposition
                                 # Temperature sensitivity: rate doubles per 10°C
                                 # Common biological Q10 value (Lecture 8 analogy)
    
    # Atmospheric exchange (based on Lecture 11 residence times)
    tau_atm_ocean = 4.0         # Atmosphere-ocean exchange timescale [years]
                                 # Lecture 11, Slide 12: Mixed layer residence ~4 years


# ============================================================================
# CLIMATE PARAMETERS
# ============================================================================

class ClimateParams:
    """
    Parameters for temperature and radiative forcing
    
    Based on:
    - Lecture 3: Energy balance equations
    - Lecture 4: Ice-albedo feedback and climate sensitivity
    """
    
    # Temperature initial condition
    T_INIT = 0.0                # Initial surface temperature [°C]
                                 # Start at reference (0°C)
    
    # Albedo parameters (Lecture 4, Slides 18-22: Ice-albedo feedback)
    alpha_min = 0.30            # Minimum albedo (ice-free)
    alpha_max = 0.60            # Maximum albedo (ice-covered)
    T_albedo_mid = -5.0         # Midpoint temperature for albedo transition [°C]
    T_albedo_width = 10.0       # Width of transition [°C]
    
    # Radiative forcing (Lecture 4, Slide 12-14)
    alpha_CO2 = 5.35           # CO2 radiative forcing coefficient [W/m²]
                                # ΔF = α_CO2 * ln(C/C0)
    
    # Climate sensitivity (derived from Lecture 3, 4)
    lambda_climate = 1.2        # Climate feedback parameter [W/(m²·K)]
                                # Relates forcing to temperature change
                                # Typical value: 1-2 W/(m²·K)
    
    # Heat capacity (Lecture 3 energy balance)
    C_heat = 50.0              # Effective heat capacity [W·year/(m²·K)]
                               # Represents thermal inertia of system
                               # Increased to slow temperature response


# ============================================================================
# ANTHROPOGENIC FORCING
# ============================================================================

class AnthropogenicParams:
    """
    Anthropogenic CO2 emissions scenarios
    
    Based on RCP scenarios and course material
    """
    
    # Emission rates [PgC/year]
    E_2020 = 10.0              # ~10 PgC/year fossil fuel emissions (2020)
    E_growth_rate = 0.005       # 0.5% per year growth (conservative scenario)
    
    # Alternative: can use step functions or RCP-like scenarios


# ============================================================================
# NUMERICAL PARAMETERS
# ============================================================================

class NumericalParams:
    """
    Parameters for numerical integration and simulation
    """
    
    # Time parameters
    t_start = 0.0              # Start time [years]
    t_end_short = 100.0        # Short simulation: decadal scale [years]
    t_end_long = 500.0         # Long simulation: century scale [years]
    dt = 0.1                   # Time step [years]
    
    # Tolerance for equilibrium detection
    equilibrium_tol = 1e-6     # Rate of change threshold


# ============================================================================
# FEEDBACK FLAGS
# ============================================================================

class FeedbackFlags:
    """
    Flags to enable/disable specific feedback mechanisms
    For sensitivity analysis and feedback quantification
    """
    
    enable_albedo_feedback = True          # Ice-albedo feedback
    enable_temp_decomposition = True       # Temperature-dependent decomposition
    enable_CO2_forcing = True              # Greenhouse forcing from CO2
    enable_permafrost_thaw = True          # Temperature-dependent thawing


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_params():
    """
    Returns a dictionary of all default parameters
    Useful for model initialization
    """
    params = {
        # Physical constants
        'sigma': SIGMA,
        'solar_constant': SOLAR_CONSTANT,
        'CO2_preind_ppm': CO2_PREINDUSTRIAL_PPM,
        'CO2_preind_PgC': CO2_PREINDUSTRIAL_PGC,
        'ppm_to_PgC': PPM_TO_PGC,
        'T_ref': T_REFERENCE,
        
        # Carbon cycle
        'C_frozen_init': CarbonParams.C_FROZEN_INIT,
        'C_active_init': CarbonParams.C_ACTIVE_INIT,
        'k_thaw_base': CarbonParams.k_thaw_base,
        'beta_thaw': CarbonParams.beta_thaw,
        'k_decomp_base': CarbonParams.k_decomp_base,
        'Q10_decomp': CarbonParams.Q10_decomp,
        'tau_atm_ocean': CarbonParams.tau_atm_ocean,
        
        # Climate
        'T_init': ClimateParams.T_INIT,
        'alpha_min': ClimateParams.alpha_min,
        'alpha_max': ClimateParams.alpha_max,
        'T_albedo_mid': ClimateParams.T_albedo_mid,
        'T_albedo_width': ClimateParams.T_albedo_width,
        'alpha_CO2': ClimateParams.alpha_CO2,
        'lambda_climate': ClimateParams.lambda_climate,
        'C_heat': ClimateParams.C_heat,
        
        # Anthropogenic
        'E_2020': AnthropogenicParams.E_2020,
        'E_growth_rate': AnthropogenicParams.E_growth_rate,
        
        # Numerical
        't_start': NumericalParams.t_start,
        't_end_short': NumericalParams.t_end_short,
        't_end_long': NumericalParams.t_end_long,
        'dt': NumericalParams.dt,
        'equilibrium_tol': NumericalParams.equilibrium_tol,
        
        # Feedback flags
        'enable_albedo_feedback': FeedbackFlags.enable_albedo_feedback,
        'enable_temp_decomposition': FeedbackFlags.enable_temp_decomposition,
        'enable_CO2_forcing': FeedbackFlags.enable_CO2_forcing,
        'enable_permafrost_thaw': FeedbackFlags.enable_permafrost_thaw,
    }
    
    return params


def print_parameter_summary():
    """
    Prints a formatted summary of all parameters with their values and units
    """
    print("="*70)
    print("ARCTIC PERMAFROST-CARBON-CLIMATE MODEL PARAMETERS")
    print("="*70)
    
    print("\n--- CARBON CYCLE ---")
    print(f"Initial frozen carbon:     {CarbonParams.C_FROZEN_INIT:.1f} PgC")
    print(f"Initial active carbon:     {CarbonParams.C_ACTIVE_INIT:.1f} PgC")
    print(f"Base thaw rate:            {CarbonParams.k_thaw_base:.3f} 1/year")
    print(f"Thaw temperature sens.:    {CarbonParams.beta_thaw:.3f} 1/°C")
    print(f"Base decomposition rate:   {CarbonParams.k_decomp_base:.3f} 1/year")
    print(f"Decomposition Q10:         {CarbonParams.Q10_decomp:.1f}")
    print(f"Atm-ocean exchange time:   {CarbonParams.tau_atm_ocean:.1f} years")
    
    print("\n--- CLIMATE ---")
    print(f"Initial temperature:       {ClimateParams.T_INIT:.1f} °C")
    print(f"Albedo range:              {ClimateParams.alpha_min:.2f} - {ClimateParams.alpha_max:.2f}")
    print(f"Albedo transition temp:    {ClimateParams.T_albedo_mid:.1f} °C")
    print(f"CO2 forcing coefficient:   {ClimateParams.alpha_CO2:.2f} W/m²")
    print(f"Climate feedback param:    {ClimateParams.lambda_climate:.2f} W/(m²·K)")
    print(f"Heat capacity:             {ClimateParams.C_heat:.1f} W·year/(m²·K)")
    
    print("\n--- ANTHROPOGENIC ---")
    print(f"2020 emission rate:        {AnthropogenicParams.E_2020:.1f} PgC/year")
    print(f"Emission growth rate:      {AnthropogenicParams.E_growth_rate:.3f} 1/year")
    
    print("\n--- NUMERICAL ---")
    print(f"Short simulation:          {NumericalParams.t_end_short:.1f} years")
    print(f"Long simulation:           {NumericalParams.t_end_long:.1f} years")
    print(f"Time step:                 {NumericalParams.dt:.2f} years")
    
    print("="*70)


if __name__ == "__main__":
    # Test parameter loading
    print_parameter_summary()
    
    # Test parameter dictionary
    params = get_default_params()
    print(f"\nTotal parameters loaded: {len(params)}")
    print("Parameter dictionary created successfully!")