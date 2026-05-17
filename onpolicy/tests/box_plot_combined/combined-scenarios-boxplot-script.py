import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"

FAULT_NAMES = {
    0: "NO_FAULT",
    1: "SINGLE_WHEEL",
    2: "DOUBLE_WHEEL",
    3: "F1: ALL_WHEEL_V0",
    4: "F2: ALL_WHEEL_V10",
    5: "F3: ALL_WHEEL_V50",
    6: "RADIAL_CAM_4",
    7: "UPFACING_CAM",
    8: "F4: PICKUP",
    9: "DROPOFF",
    10: "LASER_16",
    11: "R_COMMS",
    12: "S_COMMS"
}

def get_ordered_fault_names():
    return [FAULT_NAMES[i] for i in sorted(FAULT_NAMES.keys())]

def get_present_fault_names(df):
    present_faults = df['Fault_Name'].unique()
    ordered_faults = get_ordered_fault_names()
    return [fault for fault in ordered_faults if fault in present_faults]

def calculate_performance(file_path):
    df = pd.read_csv(file_path)
    return df['Delivery_Rate'].sum()/10/50
    # return df['Delivery_Rate'].sum()/3/20

def process_data(base_folder, scenario1, scenario2):
    results = []
    
    # Process without_mitigation data (common for both scenarios)
    baseline_folder = os.path.join(base_folder, scenario1, 'without_mitigation')
    for filename in os.listdir(baseline_folder):
        if filename.endswith('.csv'):
            parts = filename.split('_')
            n_t_part = parts[-1].split('.')[0]
            n = int(n_t_part.split('T')[0][1:])
            t = int(n_t_part.split('T')[1])
            
            baseline_file = os.path.join(baseline_folder, filename)
            p_baseline = calculate_performance(baseline_file)
            
            results.append({
                'Fault_Number': n,
                'Fault_Type': t,
                'Fault_Name': FAULT_NAMES.get(t, f"Unknown_{t}"),
                'Performance': p_baseline,
                'Condition': 'Baseline'
            })
    
    # Process with_mitigation data for both scenarios
    for scenario in [scenario1, scenario2]:
        mitigated_folder = os.path.join(base_folder, scenario, 'with_mitigation')
        for filename in os.listdir(mitigated_folder):
            if filename.endswith('.csv'):
                parts = filename.split('_')
                n_t_part = parts[-1].split('.')[0]
                n = int(n_t_part.split('T')[0][1:])
                t = int(n_t_part.split('T')[1])
                
                mitigated_file = os.path.join(mitigated_folder, filename)
                p_mitigated = calculate_performance(mitigated_file)
                
                results.append({
                    'Fault_Number': n,
                    'Fault_Type': t,
                    'Fault_Name': FAULT_NAMES.get(t, f"Unknown_{t}"),
                    'Performance': p_mitigated,
                    'Condition': f'MARL ({scenario})'
                })
    
    df_results = pd.DataFrame(results)
    
    # Add Fault_Order column
    fault_order = {name: order for order, name in enumerate(get_ordered_fault_names())}
    df_results['Fault_Order'] = df_results['Fault_Name'].map(fault_order)
    
    return df_results

def plot_combined_boxplots(df_results, scenario_pair):
    plt.figure(figsize=(22, 12))
    # sns.set_style("ticks")

    # Get only the fault names present in the data
    present_fault_names = get_present_fault_names(df_results)
    
    # Filter the dataframe to include only present fault types
    df_results_filtered = df_results[df_results['Fault_Name'].isin(present_fault_names)]
    
    # Sort the dataframe by Fault_Order and ensure Baseline comes first
    df_results_sorted = df_results_filtered.sort_values(['Fault_Order', 'Condition'])
    
    # Create custom order for conditions
    condition_order = ['Baseline'] + [cond for cond in df_results_sorted['Condition'].unique() if cond != 'Baseline']
    
    # Create the plot
    ax = sns.boxplot(x='Fault_Name', y='Performance', hue='Condition', data=df_results_sorted,
                     width=0.4, palette=sns.color_palette("pastel"),
                     order=present_fault_names, hue_order=condition_order,
                     linewidth=4, dodge=True) 
    
    # plt.title(f'Performance Comparison: Baseline vs MARL Mitigation ({scenario_pair})', fontsize=22)
    # plt.title(f'Performance of Execution in the Reduced Environment', fontsize=22)
    plt.title(f'Performance of Execution in the Standard Environment', fontsize=22)
    plt.ylabel('Performance (N-CBDS)', fontsize=20)
    plt.xlabel('Fault Type', fontsize=20)

    ax.set_xticks(range(len(present_fault_names)))
    ax.set_xticklabels(present_fault_names, rotation=0, ha='center', fontsize=18)

    
    # plt.xticks(rotation=0, ha='right', fontsize=16)
    plt.yticks(fontsize=16)

    ax.tick_params(axis='x', pad=10)  # Adjust the padding for x-axis ticks
    ax.tick_params(axis='y', pad=10)  # Adjust the padding for y-axis ticks
    
    
    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=16)
    
    # Make plot border thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Adjust layout to prevent cutting off the legend and add padding
    plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.95])
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
    
    plt.savefig(f'{scenario_pair.lower().replace("-", "_")}_performance_comparison_boxplot.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved as {scenario_pair.lower().replace('-', '_')}_performance_comparison_boxplot.png")

def main(execution_env):
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    
    if execution_env == 'RE':
        scenario1, scenario2 = 'RT-RE', 'ST-RE'
    elif execution_env == 'SE':
        scenario1, scenario2 = 'RT-SE', 'ST-SE'
    else:
        print("Invalid execution environment. Please choose 'RE' or 'SE'.")
        sys.exit(1)
    
    df_results = process_data(base_folder, scenario1, scenario2)
    plot_combined_boxplots(df_results, f"{scenario1}-{scenario2}")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {scenario1}-{scenario2}:")
    summary = df_results.groupby(['Fault_Name', 'Condition'])['Performance'].agg(['mean', 'median', 'std'])
    print(summary)
    summary.to_csv(f'{scenario1.lower()}_{scenario2.lower()}_summary_statistics.csv')
    print(f"Summary statistics saved to {scenario1.lower()}_{scenario2.lower()}_summary_statistics.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <execution_environment>")
        print("Available execution environments: RE, SE")
        sys.exit(1)
    
    execution_env = sys.argv[1].upper()
    if execution_env not in ["RE", "SE"]:
        print("Invalid execution environment. Please choose 'RE' or 'SE'.")
        sys.exit(1)
    
    main(execution_env)