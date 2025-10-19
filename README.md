# Arctic Permafrost-Carbon-Climate Feedback Model

## Overview

This project implements a coupled ordinary differential equation (ODE) system modeling the feedback between Arctic permafrost thaw, carbon release, and climate warming. The model tracks atmospheric carbon, active layer permafrost, deep permafrost, and Arctic surface temperature over time under different emission scenarios.

## Model Description

### State Variables
- **C_atm**: Atmospheric carbon (Pg C)
- **C_active**: Active layer permafrost carbon (Pg C)
- **C_deep**: Deep permafrost carbon (Pg C)
- **T_s**: Arctic surface temperature (°C)

### Key Features
- Temperature-dependent decomposition (Q10 or Arrhenius kinetics)
- Permafrost thaw threshold at 0°C with smooth sigmoid transition
- CO2 radiative forcing (logarithmic)
- Energy balance model (Budykov linearization)
- Optional ice-albedo feedback
- Three IPCC emission scenarios (RCP 2.6, 4.5, 8.5)

## Project Structure

```
permafrost_project/
├── code/
│   ├── parameters.py         # All model parameters
│   ├── helper_functions.py   # Component functions
│   ├── model.py              # Main ODE system (RHS)
│   ├── solver.py             # ODE integration
│   ├── visualization.py      # Plotting functions
│   ├── run_all.py            # Main execution script
│   └── requirements.txt      # Python dependencies
├── results/                  # Output CSV files
├── figures/                  # Output figures
└── data/                     # Input data (if any)
```

## Installation

### Requirements
- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib

### Setup

1. **Create project directory:**
   ```bash
   mkdir permafrost_project
   cd permafrost_project
   mkdir code results figures data
   cd code
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python run_all.py
```

This will:
- Run all three RCP scenarios
- Create comparison figures
- Save results to CSV files
- Display summary statistics

### Running Individual Components

#### 1. Test Parameters
```bash
python parameters.py
```
Displays all model parameters and tests emission scenarios.

#### 2. Test Helper Functions
```bash
python helper_functions.py
```
Tests decomposition rates, forcing, albedo, and thaw functions.

#### 3. Test Model RHS
```bash
python model.py
```
Tests the ODE right-hand side function and checks conservation laws.

#### 4. Test Solver
```bash
python solver.py
```
Runs a single scenario (RCP 4.5) and creates test plots.

### Custom Simulations

```python
import numpy as np
from parameters import PARAMS
from solver import run_model

# Set up initial conditions
initial_state = np.array([594.0, 174.0, 800.0, -5.0])

# Define time span (0 = year 2000)
time_points = np.linspace(0, 100, 1001)  # 2000-2100

# Run specific scenario
results = run_model(
    initial_state,
    time_points,
    PARAMS,
    forcing_scenario='RCP4.5',
    use_albedo_feedback=False
)

# Access results
print(f"Final CO2: {results['CO2_ppm'][-1]:.1f} ppm")
print(f"Final temp: {results['T_s'][-1]:.2f} °C")
```

### Multiple Scenarios

```python
from solver import run_multiple_scenarios

results_dict = run_multiple_scenarios(
    initial_state,
    time_points,
    PARAMS,
    scenarios=['RCP2.6', 'RCP4.5', 'RCP8.5'],
    use_albedo_feedback=False
)

# Compare scenarios
for scenario, res in results_dict.items():
    print(f"{scenario}: {res['T_s'][-1]:.2f}°C in 2100")
```

### Creating Plots

```python
from visualization import plot_time_series, plot_phase_portrait

# Time series comparison
fig1 = plot_time_series(results_dict, save_path='../figures/comparison.png')

# Phase portrait
fig2 = plot_phase_portrait(results['RCP4.5'], 
                           state_vars=('C_atm', 'T_s'),
                           save_path='../figures/phase.png')
```

## Key Parameters

### Initial Conditions (Year 2000)
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| C_atm_0 | 594 | Pg C | Pre-industrial (280 ppm) |
| C_active_0 | 174 | Pg C | Vonk et al. 2024 |
| C_deep_0 | 800 | Pg C | Hugelius et al. 2014 |
| T_s_0 | -5 | °C | Arctic baseline |

### Decomposition Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| Q10 | 2.6 | - | Bracho et al. 2016 |
| k_ref | 0.05 | yr⁻¹ | At 0°C |
| E_a | 67 | kJ/mol | Filimonenko & Kuzyakov 2025 |

### Climate Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| S_0 | 1370 | W/m² | Solar constant |
| A | 204 | W/m² | North et al. 1981 |
| B | 2.17 | W/(m²·°C) | North et al. 1981 |
| α_F | 5.35 | W/m² | Myhre et al. 1998 |

## Modifying Parameters

Edit `parameters.py` to change model behavior:

```python
# Increase decomposition sensitivity
Q10 = 3.0  # Default: 2.6

# Change thaw threshold
T_CRIT = -1.0  # Default: 0.0°C

# Adjust heat capacity
C_HEAT = 3.0e8  # Default: 2.08e8 J/(m²·°C)
```

## Output Files

### Results (CSV format)
- `results/rcp26_baseline.csv`
- `results/rcp45_baseline.csv`
- `results/rcp85_baseline.csv`
- `results/rcp85_with_feedback.csv`

Columns:
- `year`: Calendar year
- `time_since_2000`: Years since 2000
- `C_atm_PgC`: Atmospheric carbon
- `C_active_PgC`: Active layer carbon
- `C_deep_PgC`: Deep permafrost carbon
- `T_surface_C`: Surface temperature
- `CO2_ppm`: CO2 concentration

### Figures (PNG format)
- `figures/fig1_timeseries.png`: Multi-panel time series
- `figures/fig2_phase_portrait.png`: Phase space trajectory
- `figures/fig3_carbon_budget.png`: Stacked carbon pools
- `figures/fig4_feedback_comparison.png`: With/without albedo feedback

## Validation Checks

The model includes built-in validation:

1. **Carbon conservation**: Total carbon change matches external inputs
2. **Energy balance consistency**: Temperature responds correctly to forcing
3. **Physical constraints**: No negative carbon stocks
4. **Trend validation**: CO2 and temperature increase under warming

Run tests with:
```bash
python model.py
python solver.py
```

## Expected Results

### RCP 4.5 (Moderate Mitigation)
- **2100 CO2**: ~520-540 ppm
- **2100 Temperature**: +2-3°C above 2000
- **Permafrost carbon loss**: ~10-15% of initial stock

### RCP 8.5 (High Emissions)
- **2100 CO2**: ~900-950 ppm
- **2100 Temperature**: +5-7°C above 2000
- **Permafrost carbon loss**: ~20-30% of initial stock

## Common Issues

### Issue: Solver diverges or produces negative values
**Solution**: 
- Reduce time step: `time_points = np.linspace(0, 100, 5000)`
- Increase solver tolerances: `rtol=1e-8, atol=1e-10`
- Check parameter values are physically reasonable

### Issue: Figures not saving
**Solution**:
- Ensure `figures/` directory exists
- Check file permissions
- Try absolute paths

### Issue: Import errors
**Solution**:
```bash
pip install --upgrade numpy scipy matplotlib
```

## Extending the Model

### Adding Regional Structure
Split permafrost into Siberia, Alaska, Canada with different parameters.

### Multi-Pool Carbon
Separate active layer into fast/slow decomposition pools.

### Methane Emissions
Add CH4 production with anaerobic decomposition pathways.

### Abrupt Thaw
Include thermokarst lake formation as discrete events.

## References

### Key Papers
1. Hugelius et al. (2014) - Permafrost carbon stocks
2. Vonk et al. (2024) - Arctic carbon cycle synthesis
3. Nitzbon et al. (2024) - Tipping point assessment
4. Filimonenko & Kuzyakov (2025) - Decomposition kinetics
5. North et al. (1981) - Energy balance models
6. Myhre et al. (1998) - CO2 radiative forcing

### Course Connection
This model applies concepts from "Modeling Earth Systems: Quantitative Reasoning & Python":
- Systems thinking and feedback loops
- Coupled ODEs (Lotka-Volterra analogy)
- Energy balance models (Budykov linearization)
- Nonlinear dynamics (threshold behavior)
- Numerical methods (scipy.integrate.odeint)

## Authors

**Course Project**: Modeling Earth Systems: Quantitative Reasoning & Python

## License

Educational use. See course materials for attribution requirements.

## Support

For questions or issues:
1. Check this README
2. Review code comments and docstrings
3. Test individual components
4. Consult course materials

---

**Last Updated**: 2025