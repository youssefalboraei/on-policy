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

def calculate_performance(file_path, scenario):
    df = pd.read_csv(file_path)
    if 'SE' in scenario:
        return df['Delivery_Rate'].sum() / 10 / 50  # Standard Environment
    else:
        return df['Delivery_Rate'].sum() / 3 / 20   # Reduced Environment

def process_data(base_folder):
    results = []
    scenarios = ['RT-RE', 'ST-RE', 'RT-SE', 'ST-SE']
    
    for scenario in scenarios:
        baseline_folder = os.path.join(base_folder, scenario, 'without_mitigation')
        mitigated_folder = os.path.join(base_folder, scenario, 'with_mitigation')
        
        # Process without_mitigation data
        for filename in os.listdir(baseline_folder):
            if filename.endswith('.csv'):
                parts = filename.split('_')
                n_t_part = parts[-1].split('.')[0]
                n = int(n_t_part.split('T')[0][1:])
                t = int(n_t_part.split('T')[1])
                
                baseline_file = os.path.join(baseline_folder, filename)
                p_baseline = calculate_performance(baseline_file, scenario)
                
                results.append({
                    'Fault_Number': n,
                    'Fault_Type': t,
                    'Fault_Name': FAULT_NAMES.get(t, f"Unknown_{t}"),
                    'Performance': p_baseline,
                    'Condition': f'Baseline ({scenario})',
                    'Scenario': scenario
                })
        
        # Process with_mitigation data
        for filename in os.listdir(mitigated_folder):
            if filename.endswith('.csv'):
                parts = filename.split('_')
                n_t_part = parts[-1].split('.')[0]
                n = int(n_t_part.split('T')[0][1:])
                t = int(n_t_part.split('T')[1])
                
                mitigated_file = os.path.join(mitigated_folder, filename)
                p_mitigated = calculate_performance(mitigated_file, scenario)
                
                results.append({
                    'Fault_Number': n,
                    'Fault_Type': t,
                    'Fault_Name': FAULT_NAMES.get(t, f"Unknown_{t}"),
                    'Performance': p_mitigated,
                    'Condition': f'MARL ({scenario})',
                    'Scenario': scenario
                })
    
    df_results = pd.DataFrame(results)
    
    # Add Fault_Order column
    fault_order = {name: order for order, name in enumerate(get_ordered_fault_names())}
    df_results['Fault_Order'] = df_results['Fault_Name'].map(fault_order)
    
    return df_results

# def plot_combined_boxplots(df_results):
#     plt.figure(figsize=(24, 14))

#     # Get only the fault names present in the data
#     present_fault_names = get_present_fault_names(df_results)
    
#     # Filter the dataframe to include only present fault types
#     df_results_filtered = df_results[df_results['Fault_Name'].isin(present_fault_names)]
    
#     # Sort the dataframe by Fault_Order
#     df_results_sorted = df_results_filtered.sort_values(['Fault_Order', 'Condition'])
    
#     # Create custom order for conditions
#     condition_order = [f'Baseline ({s})' for s in ['RT-RE', 'ST-RE', 'RT-SE', 'ST-SE']] + \
#                       [f'MARL ({s})' for s in ['RT-RE', 'ST-RE', 'RT-SE', 'ST-SE']]
    
#     # Create the plot
#     ax = sns.boxplot(x='Fault_Name', y='Performance', hue='Condition', data=df_results_sorted,
#                      width=0.8, palette=sns.color_palette("pastel", n_colors=8),
#                      order=present_fault_names, hue_order=condition_order,
#                      linewidth=2, dodge=True)
    
#     plt.title('Performance Comparison: Baseline vs MARL Mitigation (All Scenarios)', fontsize=22)
#     plt.ylabel('Performance (N-CBDS)', fontsize=20)
#     plt.xlabel('Fault Type', fontsize=20)

#     ax.set_xticks(range(len(present_fault_names)))
#     ax.set_xticklabels(present_fault_names, rotation=45, ha='right', fontsize=16)
#     plt.yticks(fontsize=16)

#     ax.tick_params(axis='x', pad=10)
#     ax.tick_params(axis='y', pad=10)
    
#     # Move legend outside the plot
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=14)
    
#     # Make plot border thicker
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
#         spine.set_color('black')
    
#     plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.95])
    
#     plt.savefig('all_scenarios_performance_comparison_boxplot.png', dpi=600, bbox_inches='tight')
#     plt.close()
#     print("Box plot saved as all_scenarios_performance_comparison_boxplot.png")

def plot_combined_boxplots(df_results):
    plt.figure(figsize=(24, 14))

    # Get only the fault names present in the data
    present_fault_names = get_present_fault_names(df_results)
    
    # Filter the dataframe to include only present fault types
    df_results_filtered = df_results[df_results['Fault_Name'].isin(present_fault_names)]
    
    # Sort the dataframe by Fault_Order
    df_results_sorted = df_results_filtered.sort_values(['Fault_Order', 'Condition'])
    
    # Create custom order for conditions
    condition_order = ['Baseline (RT-RE)', 'MARL (RT-RE)', 'Baseline (ST-RE)', 'MARL (ST-RE)',
                       'Baseline (RT-SE)', 'MARL (RT-SE)', 'Baseline (ST-SE)', 'MARL (ST-SE)']
    
    # Define colors for different conditions
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a',  # Blues and greens for RE
              '#ff7f0e', '#ffbb78', '#d62728', '#ff9896']  # Oranges and reds for SE
    
    # Create the plot
    ax = sns.boxplot(x='Fault_Name', y='Performance', hue='Condition', data=df_results_sorted,
                     width=0.7, palette=colors,
                     order=present_fault_names, hue_order=condition_order,
                     linewidth=2)
    
    plt.title('Performance Comparison: Baseline vs MARL Mitigation (RE and SE)', fontsize=22)
    plt.ylabel('Performance (N-CBDS)', fontsize=20)
    plt.xlabel('Fault Type', fontsize=20)

    # Adjust x-axis to add space between groups
    num_faults = len(present_fault_names)
    ax.set_xticks(np.arange(num_faults) * 1.2)
    ax.set_xticklabels(present_fault_names, rotation=0, ha='center', fontsize=16)
    ax.set_xlim(-0.5, num_faults * 1.2 - 0.5)

    plt.yticks(fontsize=16)

    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
    
    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Condition', title_fontsize='16', fontsize=14,
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    # Add vertical lines to separate fault types
    for i in range(1, num_faults):
        ax.axvline(x=i * 1.2 - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add a horizontal line to separate RE and SE
    y_min, y_max = ax.get_ylim()
    ax.axhline(y=(y_min + y_max) / 2, color='gray', linestyle='--', linewidth=2)
    
    # Add RE and SE labels
    ax.text(-0.5, y_max, 'Reduced Environment (RE)', fontsize=16, va='top', ha='left', weight='bold')
    ax.text(-0.5, y_min, 'Standard Environment (SE)', fontsize=16, va='bottom', ha='left', weight='bold')
    
    # Make plot border thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.95])
    
    plt.savefig('all_scenarios_performance_comparison_boxplot_grouped_spaced.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("Box plot saved as all_scenarios_performance_comparison_boxplot_grouped_spaced.png")
    
def main():
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    
    df_results = process_data(base_folder)
    plot_combined_boxplots(df_results)
    
    # Print summary statistics
    print("\nSummary Statistics for all scenarios:")
    summary = df_results.groupby(['Fault_Name', 'Condition'])['Performance'].agg(['mean', 'median', 'std'])
    print(summary)
    summary.to_csv('all_scenarios_summary_statistics.csv')
    print("Summary statistics saved to all_scenarios_summary_statistics.csv")

if __name__ == "__main__":
    main()