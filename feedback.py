"""
Phase 3: Feedback Quantification

This module provides tools for:
1. Running all feedback combinations
2. Calculating amplification factors
3. Creating comparison tables
4. Quantifying individual feedback contributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model
import parameters as params
import visualization as viz


# ============================================================================
# FEEDBACK SCENARIO GENERATION
# ============================================================================

def generate_all_feedback_scenarios():
    """
    Generate all 16 combinations of feedback on/off states
    
    Feedbacks:
    1. Albedo feedback
    2. Temperature-dependent decomposition
    3. CO2 greenhouse forcing
    4. Permafrost thaw
    
    Returns:
    --------
    scenarios : dict
        Dictionary of scenario name -> parameter dictionary
    """
    # Base parameters
    base_params = params.get_default_params()
    
    scenarios = {}
    
    # Feedback flags
    feedbacks = [
        ('albedo', 'enable_albedo_feedback'),
        ('decomp', 'enable_temp_decomposition'),
        ('CO2', 'enable_CO2_forcing'),
        ('thaw', 'enable_permafrost_thaw')
    ]
    
    # Generate all 2^4 = 16 combinations
    for i in range(2**4):
        # Binary representation determines which feedbacks are on
        binary = format(i, '04b')
        
        # Create parameter dict for this scenario
        scenario_params = base_params.copy()
        
        # Build scenario name and set flags
        scenario_name_parts = []
        for idx, (name, flag) in enumerate(feedbacks):
            is_enabled = (binary[idx] == '1')
            scenario_params[flag] = is_enabled
            
            if is_enabled:
                scenario_name_parts.append(name)
        
        if len(scenario_name_parts) == 0:
            scenario_name = "No Feedbacks"
        elif len(scenario_name_parts) == 4:
            scenario_name = "All Feedbacks"
        else:
            scenario_name = " + ".join(scenario_name_parts)
        
        scenarios[scenario_name] = scenario_params
    
    return scenarios


def generate_key_feedback_scenarios():
    """
    Generate key feedback scenarios for focused analysis
    
    Returns:
    --------
    scenarios : dict
        Dictionary of scenario name -> parameter dictionary
    """
    base_params = params.get_default_params()
    
    scenarios = {
        # Baseline
        'All Feedbacks': base_params.copy(),
        
        # Isolate each feedback (all on except one)
        'No Albedo': {**base_params, 'enable_albedo_feedback': False},
        'No Temp Decomp': {**base_params, 'enable_temp_decomposition': False},
        'No CO2 Forcing': {**base_params, 'enable_CO2_forcing': False},
        'No Permafrost Thaw': {**base_params, 'enable_permafrost_thaw': False},
        
        # Individual feedbacks (only one on)
        'Only Albedo': {
            **base_params,
            'enable_albedo_feedback': True,
            'enable_temp_decomposition': False,
            'enable_CO2_forcing': False,
            'enable_permafrost_thaw': False
        },
        'Only Temp Decomp': {
            **base_params,
            'enable_albedo_feedback': False,
            'enable_temp_decomposition': True,
            'enable_CO2_forcing': False,
            'enable_permafrost_thaw': False
        },
        'Only CO2 Forcing': {
            **base_params,
            'enable_albedo_feedback': False,
            'enable_temp_decomposition': False,
            'enable_CO2_forcing': True,
            'enable_permafrost_thaw': False
        },
        'Only Permafrost Thaw': {
            **base_params,
            'enable_albedo_feedback': False,
            'enable_temp_decomposition': False,
            'enable_CO2_forcing': False,
            'enable_permafrost_thaw': True
        },
        
        # No feedbacks
        'No Feedbacks': {
            **base_params,
            'enable_albedo_feedback': False,
            'enable_temp_decomposition': False,
            'enable_CO2_forcing': False,
            'enable_permafrost_thaw': False
        }
    }
    
    return scenarios


# ============================================================================
# FEEDBACK QUANTIFICATION
# ============================================================================

def quantify_feedbacks(scenarios, t_span=(0, 100), dt=0.5):
    """
    Run all scenarios and quantify feedback contributions
    
    Parameters:
    -----------
    scenarios : dict
        Dictionary of scenario name -> parameter dictionary
    t_span : tuple
        Time span for simulations
    dt : float
        Time step
    
    Returns:
    --------
    results : dict
        Dictionary of results with:
        - 'simulations': dict of (t, solution) for each scenario
        - 'final_states': DataFrame of final states
        - 'changes': DataFrame of changes from initial
        - 'amplification': DataFrame of amplification factors
    """
    print(f"Running {len(scenarios)} feedback scenarios...")
    
    # Run all simulations
    simulations = {}
    for scenario_name, params_dict in scenarios.items():
        print(f"  Running: {scenario_name}")
        t, sol = model.run_simulation(t_span, params_dict=params_dict, dt=dt)
        simulations[scenario_name] = (t, sol)
    
    # Extract final states
    final_states = {}
    for scenario_name, (t, sol) in simulations.items():
        final_states[scenario_name] = {
            'C_frozen': sol[-1, 0],
            'C_active': sol[-1, 1],
            'C_atm': sol[-1, 2],
            'T_s': sol[-1, 3],
            'time': t[-1]
        }
    
    # Convert to DataFrame
    df_final = pd.DataFrame(final_states).T
    
    # Calculate changes from initial
    initial_state = simulations[list(simulations.keys())[0]][1][0, :]
    
    changes = {}
    for scenario_name, (t, sol) in simulations.items():
        changes[scenario_name] = {
            'ΔC_frozen': sol[-1, 0] - initial_state[0],
            'ΔC_active': sol[-1, 1] - initial_state[1],
            'ΔC_atm': sol[-1, 2] - initial_state[2],
            'ΔT_s': sol[-1, 3] - initial_state[3],
        }
    
    df_changes = pd.DataFrame(changes).T
    
    # Calculate amplification factors (relative to no feedbacks baseline)
    if 'No Feedbacks' in simulations:
        baseline_T = simulations['No Feedbacks'][1][-1, 3]
        baseline_C = simulations['No Feedbacks'][1][-1, 2]
        
        amplification = {}
        for scenario_name, (t, sol) in simulations.items():
            if scenario_name != 'No Feedbacks':
                T_final = sol[-1, 3]
                C_final = sol[-1, 2]
                
                amplification[scenario_name] = {
                    'T_amplification': T_final / baseline_T if abs(baseline_T) > 0.01 else np.nan,
                    'C_amplification': C_final / baseline_C if baseline_C > 0 else np.nan,
                    'T_difference': T_final - baseline_T,
                    'C_difference': C_final - baseline_C
                }
        
        df_amplification = pd.DataFrame(amplification).T
    else:
        df_amplification = None
    
    results = {
        'simulations': simulations,
        'final_states': df_final,
        'changes': df_changes,
        'amplification': df_amplification,
        'initial_state': initial_state
    }
    
    return results


def calculate_feedback_contributions(results):
    """
    Calculate individual feedback contributions
    
    Uses the approach of comparing:
    - All feedbacks vs. All except one (removal effect)
    - Only one feedback vs. No feedbacks (individual effect)
    
    Parameters:
    -----------
    results : dict
        Results from quantify_feedbacks
    
    Returns:
    --------
    contributions : dict
        Dictionary with 'removal_effects' and 'individual_effects' DataFrames
    """
    simulations = results['simulations']
    
    # Get reference values
    if 'All Feedbacks' in simulations:
        all_fb_T = simulations['All Feedbacks'][1][-1, 3]
        all_fb_C = simulations['All Feedbacks'][1][-1, 2]
    else:
        all_fb_T = all_fb_C = None
    
    if 'No Feedbacks' in simulations:
        no_fb_T = simulations['No Feedbacks'][1][-1, 3]
        no_fb_C = simulations['No Feedbacks'][1][-1, 2]
    else:
        no_fb_T = no_fb_C = None
    
    # Calculate removal effects (importance of each feedback)
    removal_effects = {}
    feedback_removals = {
        'Albedo': 'No Albedo',
        'Temp Decomp': 'No Temp Decomp',
        'CO2 Forcing': 'No CO2 Forcing',
        'Permafrost Thaw': 'No Permafrost Thaw'
    }
    
    for fb_name, scenario_name in feedback_removals.items():
        if scenario_name in simulations and all_fb_T is not None:
            removed_T = simulations[scenario_name][1][-1, 3]
            removed_C = simulations[scenario_name][1][-1, 2]
            
            removal_effects[fb_name] = {
                'ΔT_when_removed': all_fb_T - removed_T,
                'ΔC_when_removed': all_fb_C - removed_C,
                'T_contribution_%': 100 * (all_fb_T - removed_T) / all_fb_T if all_fb_T != 0 else 0,
                'final_T_without': removed_T,
                'final_C_without': removed_C
            }
    
    df_removal = pd.DataFrame(removal_effects).T if removal_effects else None
    
    # Calculate individual effects (effect of each feedback alone)
    individual_effects = {}
    feedback_only = {
        'Albedo': 'Only Albedo',
        'Temp Decomp': 'Only Temp Decomp',
        'CO2 Forcing': 'Only CO2 Forcing',
        'Permafrost Thaw': 'Only Permafrost Thaw'
    }
    
    for fb_name, scenario_name in feedback_only.items():
        if scenario_name in simulations and no_fb_T is not None:
            only_T = simulations[scenario_name][1][-1, 3]
            only_C = simulations[scenario_name][1][-1, 2]
            
            individual_effects[fb_name] = {
                'ΔT_alone': only_T - no_fb_T,
                'ΔC_alone': only_C - no_fb_C,
                'final_T_alone': only_T,
                'final_C_alone': only_C
            }
    
    df_individual = pd.DataFrame(individual_effects).T if individual_effects else None
    
    contributions = {
        'removal_effects': df_removal,
        'individual_effects': df_individual
    }
    
    return contributions


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feedback_comparison(results, variable='T_s', filename=None):
    """
    Create bar chart comparing final states across scenarios
    
    Parameters:
    -----------
    results : dict
        Results from quantify_feedbacks
    variable : str
        Which variable to plot ('T_s', 'C_atm', 'C_frozen', 'C_active')
    filename : str, optional
        Save filename
    """
    viz.set_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get data
    df = results['final_states']
    scenario_names = df.index.tolist()
    values = df[variable].values
    
    # Sort by value
    sorted_indices = np.argsort(values)
    scenario_names = [scenario_names[i] for i in sorted_indices]
    values = values[sorted_indices]
    
    # Color coding
    colors = []
    for name in scenario_names:
        if 'All Feedbacks' in name:
            colors.append('darkgreen')
        elif 'No Feedbacks' in name:
            colors.append('darkred')
        elif 'Only' in name:
            colors.append('orange')
        elif 'No' in name:
            colors.append('lightblue')
        else:
            colors.append('gray')
    
    # Create bar chart
    bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(scenario_names, fontsize=10)
    
    # Labels
    var_labels = {
        'T_s': 'Temperature [°C]',
        'C_atm': 'Atmospheric CO₂ [PgC]',
        'C_frozen': 'Frozen Carbon [PgC]',
        'C_active': 'Active Carbon [PgC]'
    }
    
    ax.set_xlabel(var_labels.get(variable, variable), fontsize=13)
    ax.set_title(f'Final {var_labels.get(variable, variable)} by Scenario', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f' {val:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


def plot_contribution_analysis(contributions, filename=None):
    """
    Plot feedback contribution analysis
    
    Parameters:
    -----------
    contributions : dict
        Contributions from calculate_feedback_contributions
    filename : str, optional
        Save filename
    """
    viz.set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feedback Contribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Removal effects (temperature)
    if contributions['removal_effects'] is not None:
        df = contributions['removal_effects']
        ax = axes[0, 0]
        
        feedbacks = df.index.tolist()
        values = df['ΔT_when_removed'].values
        
        bars = ax.bar(range(len(feedbacks)), values, color='red', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(feedbacks)))
        ax.set_xticklabels(feedbacks, rotation=45, ha='right')
        ax.set_ylabel('Temperature Reduction [°C]', fontsize=11)
        ax.set_title('Effect of Removing Each Feedback', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}°C',
                   ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Individual effects (temperature)
    if contributions['individual_effects'] is not None:
        df = contributions['individual_effects']
        ax = axes[0, 1]
        
        feedbacks = df.index.tolist()
        values = df['ΔT_alone'].values
        
        bars = ax.bar(range(len(feedbacks)), values, color='green', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(feedbacks)))
        ax.set_xticklabels(feedbacks, rotation=45, ha='right')
        ax.set_ylabel('Temperature Increase [°C]', fontsize=11)
        ax.set_title('Effect of Each Feedback Alone', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}°C',
                   ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Percentage contributions
    if contributions['removal_effects'] is not None:
        df = contributions['removal_effects']
        ax = axes[1, 0]
        
        feedbacks = df.index.tolist()
        values = df['T_contribution_%'].values
        
        bars = ax.bar(range(len(feedbacks)), values, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(feedbacks)))
        ax.set_xticklabels(feedbacks, rotation=45, ha='right')
        ax.set_ylabel('Contribution [%]', fontsize=11)
        ax.set_title('Percentage Contribution to Total Warming', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Final temperatures comparison
    if contributions['removal_effects'] is not None and contributions['individual_effects'] is not None:
        ax = axes[1, 1]
        
        df_rem = contributions['removal_effects']
        df_ind = contributions['individual_effects']
        
        feedbacks = df_rem.index.tolist()
        without_values = df_rem['final_T_without'].values
        alone_values = df_ind['final_T_alone'].values
        
        x = np.arange(len(feedbacks))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, without_values, width, label='Without this feedback',
                      color='lightblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, alone_values, width, label='Only this feedback',
                      color='lightgreen', alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(feedbacks, rotation=45, ha='right')
        ax.set_ylabel('Final Temperature [°C]', fontsize=11)
        ax.set_title('Final Temperature: Without vs. Only Each Feedback', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    return fig


# ============================================================================
# TABLE GENERATION
# ============================================================================

def create_comparison_table(results, contributions, filename=None):
    """
    Create comprehensive comparison table
    
    Parameters:
    -----------
    results : dict
        Results from quantify_feedbacks
    contributions : dict
        Contributions from calculate_feedback_contributions
    filename : str, optional
        If provided, save table to CSV
    
    Returns:
    --------
    table : DataFrame
        Comprehensive comparison table
    """
    # Combine all information
    df_final = results['final_states']
    df_changes = results['changes']
    
    # Select key scenarios
    key_scenarios = ['All Feedbacks', 'No Feedbacks', 
                    'No Albedo', 'No Temp Decomp', 'No CO2 Forcing', 'No Permafrost Thaw',
                    'Only Albedo', 'Only Temp Decomp', 'Only CO2 Forcing', 'Only Permafrost Thaw']
    
    # Filter to key scenarios that exist
    key_scenarios = [s for s in key_scenarios if s in df_final.index]
    
    # Create summary table
    table = pd.DataFrame({
        'Final T [°C]': df_final.loc[key_scenarios, 'T_s'],
        'ΔT [°C]': df_changes.loc[key_scenarios, 'ΔT_s'],
        'Final C_atm [PgC]': df_final.loc[key_scenarios, 'C_atm'],
        'ΔC_atm [PgC]': df_changes.loc[key_scenarios, 'ΔC_atm'],
        'Final C_frozen [PgC]': df_final.loc[key_scenarios, 'C_frozen'],
        'ΔC_frozen [PgC]': df_changes.loc[key_scenarios, 'ΔC_frozen'],
    })
    
    if filename:
        table.to_csv(filename)
        print(f"Table saved: {filename}")
    
    return table


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing feedback quantification module...\n")
    
    # Generate scenarios
    print("Generating feedback scenarios...")
    scenarios = generate_key_feedback_scenarios()
    print(f"Created {len(scenarios)} scenarios\n")
    
    # Run quantification (short test)
    print("Running feedback quantification (20-year test)...")
    results = quantify_feedbacks(scenarios, t_span=(0, 20), dt=0.5)
    
    print("\nFinal States:")
    print(results['final_states'][['T_s', 'C_atm']])
    
    print("\n✓ Feedback quantification module ready!")