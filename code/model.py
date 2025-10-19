"""
model.py
Arctic Permafrost-Carbon-Climate Feedback Model
Main ODE system definition (right-hand side function)
"""

import numpy as np
from helper_functions import (
    decomposition_rate_q10,
    radiative_forcing_co2,
    thaw_flux,
    carbon_uptake,
    incoming_solar,
    outgoing_longwave
)


def permafrost_model(state, t, params, forcing_func, use_albedo_feedback=False):
    """
    Right-hand side (RHS) of the coupled ODE system
    
    System of 4 equations:
        dC_atm/dt    = F_anthro + F_decomp - F_uptake
        dC_active/dt = F_thaw - F_decomp
        dC_deep/dt   = -F_thaw
        dT_s/dt      = (S_in - OLR + ΔF_CO2) / C_heat
    
    Parameters:
        state : array-like, shape (4,)
            Current state [C_atm, C_active, C_deep, T_s]
        t : float
            Current time [years since 2000]
        params : dict
            Model parameters
        forcing_func : function
            Anthropogenic emissions function F(t)
        use_albedo_feedback : bool
            Whether to include ice-albedo feedback
    
    Returns:
        derivatives : numpy array, shape (4,)
            Time derivatives [dC_atm/dt, dC_active/dt, dC_deep/dt, dT_s/dt]
    """
    
    # Unpack state variables
    C_atm, C_active, C_deep, T_s = state
    
    # =========================================================================
    # CARBON FLUXES
    # =========================================================================
    
    # Anthropogenic emissions (prescribed by scenario)
    F_anthro = forcing_func(t)
    
    # Decomposition flux (active layer → atmosphere)
    k_decomp = decomposition_rate_q10(T_s, params)
    F_decomp = k_decomp * C_active
    
    # Carbon uptake by ocean and land
    F_uptake = carbon_uptake(C_atm, params)
    
    # Thaw flux (deep permafrost → active layer)
    F_thaw = thaw_flux(T_s, C_deep, params)
    
    # =========================================================================
    # ENERGY BALANCE
    # =========================================================================
    
    # Incoming solar radiation (albedo-dependent)
    S_in = incoming_solar(T_s, params, use_albedo_feedback)
    
    # Outgoing longwave radiation (Budykov)
    OLR = outgoing_longwave(T_s, params)
    
    # CO2 radiative forcing
    delta_F_co2 = radiative_forcing_co2(C_atm, params)
    
    # Heat capacity
    C_heat = params['C_heat']
    
    # =========================================================================
    # TIME DERIVATIVES
    # =========================================================================
    
    # Atmospheric carbon balance
    dC_atm_dt = F_anthro + F_decomp - F_uptake
    
    # Active layer carbon balance
    dC_active_dt = F_thaw - F_decomp
    
    # Deep permafrost carbon balance
    dC_deep_dt = -F_thaw
    
    # Surface temperature evolution
    # Convert W/m² to temperature change rate
    # Need to convert from W/m² to °C/year
    # 1 W/m² = 1 J/(s·m²)
    # 1 year = 3.156e7 s
    energy_imbalance = S_in - OLR + delta_F_co2  # [W/m²]
    dT_s_dt = (energy_imbalance / C_heat) * params['PgC_to_ppm']  # Crude conversion
    
    # More accurate unit conversion:
    # [W/m²] / [J/(m²·°C)] = [J/(s·m²)] / [J/(m²·°C)] = [°C/s]
    # Multiply by seconds per year to get [°C/yr]
    SEC_PER_YEAR = 3.156e7
    dT_s_dt = (energy_imbalance / C_heat) * SEC_PER_YEAR  # [°C/yr]
    
    # Return derivatives as numpy array
    derivatives = np.array([dC_atm_dt, dC_active_dt, dC_deep_dt, dT_s_dt])
    
    return derivatives


def permafrost_model_with_diagnostics(state, t, params, forcing_func, 
                                      use_albedo_feedback=False):
    """
    Extended version that returns both derivatives and diagnostic information
    
    Useful for debugging and understanding model behavior
    
    Returns:
        derivatives : numpy array
            Time derivatives
        diagnostics : dict
            Dictionary containing all intermediate fluxes and values
    """
    
    # Unpack state
    C_atm, C_active, C_deep, T_s = state
    
    # Calculate all components
    F_anthro = forcing_func(t)
    k_decomp = decomposition_rate_q10(T_s, params)
    F_decomp = k_decomp * C_active
    F_uptake = carbon_uptake(C_atm, params)
    F_thaw = thaw_flux(T_s, C_deep, params)
    
    S_in = incoming_solar(T_s, params, use_albedo_feedback)
    OLR = outgoing_longwave(T_s, params)
    delta_F_co2 = radiative_forcing_co2(C_atm, params)
    
    # Derivatives
    dC_atm_dt = F_anthro + F_decomp - F_uptake
    dC_active_dt = F_thaw - F_decomp
    dC_deep_dt = -F_thaw
    
    SEC_PER_YEAR = 3.156e7
    C_heat = params['C_heat']
    energy_imbalance = S_in - OLR + delta_F_co2
    dT_s_dt = (energy_imbalance / C_heat) * SEC_PER_YEAR
    
    derivatives = np.array([dC_atm_dt, dC_active_dt, dC_deep_dt, dT_s_dt])
    
    # Diagnostics dictionary
    diagnostics = {
        'time': t,
        'F_anthro': F_anthro,
        'F_decomp': F_decomp,
        'F_uptake': F_uptake,
        'F_thaw': F_thaw,
        'k_decomp': k_decomp,
        'S_in': S_in,
        'OLR': OLR,
        'delta_F_co2': delta_F_co2,
        'energy_imbalance': energy_imbalance,
        'net_carbon_flux': dC_atm_dt,
    }
    
    return derivatives, diagnostics


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the RHS function with initial conditions"""
    from parameters import PARAMS, SCENARIOS
    
    print("=" * 70)
    print("TESTING PERMAFROST MODEL RHS FUNCTION")
    print("=" * 70)
    
    # Initial state
    state_0 = np.array([
        PARAMS['C_atm_0'],
        PARAMS['C_active_0'],
        PARAMS['C_deep_0'],
        PARAMS['T_s_0']
    ])
    
    print("\nINITIAL STATE (Year 2000):")
    print(f"  C_atm:    {state_0[0]:.1f} Pg C")
    print(f"  C_active: {state_0[1]:.1f} Pg C")
    print(f"  C_deep:   {state_0[2]:.1f} Pg C")
    print(f"  T_s:      {state_0[3]:.1f} °C")
    
    # Test with RCP 4.5 at t=0
    print("\n" + "=" * 70)
    print("TEST 1: Initial derivatives (t=0, RCP 4.5)")
    print("=" * 70)
    
    forcing = SCENARIOS['RCP4.5']
    derivatives, diagnostics = permafrost_model_with_diagnostics(
        state_0, 0.0, PARAMS, forcing, use_albedo_feedback=False
    )
    
    print("\nFLUXES:")
    print(f"  Anthropogenic emissions: {diagnostics['F_anthro']:.3f} Pg C/yr")
    print(f"  Decomposition:           {diagnostics['F_decomp']:.3f} Pg C/yr")
    print(f"  Uptake:                  {diagnostics['F_uptake']:.3f} Pg C/yr")
    print(f"  Thaw:                    {diagnostics['F_thaw']:.3f} Pg C/yr")
    print(f"  Decomp rate constant:    {diagnostics['k_decomp']:.4f} yr^-1")
    
    print("\nENERGY BALANCE:")
    print(f"  Incoming solar:          {diagnostics['S_in']:.2f} W/m²")
    print(f"  Outgoing longwave:       {diagnostics['OLR']:.2f} W/m²")
    print(f"  CO2 forcing:             {diagnostics['delta_F_co2']:.2f} W/m²")
    print(f"  Energy imbalance:        {diagnostics['energy_imbalance']:.2f} W/m²")
    
    print("\nDERIVATIVES:")
    print(f"  dC_atm/dt:    {derivatives[0]:+.3f} Pg C/yr")
    print(f"  dC_active/dt: {derivatives[1]:+.3f} Pg C/yr")
    print(f"  dC_deep/dt:   {derivatives[2]:+.3f} Pg C/yr")
    print(f"  dT_s/dt:      {derivatives[3]:+.4f} °C/yr")
    
    # Sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS:")
    print("=" * 70)
    
    # Check 1: Carbon conservation
    total_carbon_change = derivatives[0] + derivatives[1] + derivatives[2]
    external_input = diagnostics['F_anthro'] - diagnostics['F_uptake']
    print(f"\n1. CARBON CONSERVATION:")
    print(f"   Total system C change: {total_carbon_change:.3f} Pg C/yr")
    print(f"   External net input:    {external_input:.3f} Pg C/yr")
    print(f"   Difference:            {abs(total_carbon_change - external_input):.6f}")
    print(f"   ✓ PASS" if abs(total_carbon_change - external_input) < 1e-6 else "   ✗ FAIL")
    
    # Check 2: Temperature should increase with positive energy imbalance
    print(f"\n2. ENERGY BALANCE SIGN:")
    print(f"   Energy imbalance: {diagnostics['energy_imbalance']:.2f} W/m²")
    print(f"   dT/dt:            {derivatives[3]:.4f} °C/yr")
    consistent = (diagnostics['energy_imbalance'] > 0 and derivatives[3] > 0) or \
                 (diagnostics['energy_imbalance'] < 0 and derivatives[3] < 0)
    print(f"   ✓ PASS (signs consistent)" if consistent else "   ✗ FAIL (sign mismatch)")
    
    # Check 3: Atmospheric CO2 should increase
    print(f"\n3. ATMOSPHERIC CARBON TREND:")
    print(f"   dC_atm/dt:        {derivatives[0]:.3f} Pg C/yr")
    print(f"   ✓ PASS (positive, as expected)" if derivatives[0] > 0 else "   ✗ WARNING (negative)")
    
    print("\n" + "=" * 70)