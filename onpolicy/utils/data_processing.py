import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for all plots
plt.style.use('seaborn-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"

FAULT_NAMES = {
    0: "DEFAULT",
    1: "SINGLE_WHEEL",
    2: "DOUBLE_WHEEL",
    3: "ALL_WHEEL_V0",
    4: "ALL_WHEEL_V10",
    5: "ALL_WHEEL_V50",
    6: "RADIAL_CAM_4",
    7: "UPFACING_CAM",
    8: "PICKUP",
    9: "DROPOFF",
    10: "LASER_16",
    11: "R_COMMS",
    12: "S_COMMS"
}

def calculate_performance(file_path):
    df = pd.read_csv(file_path)
    return df['Delivery_Rate'].sum()

def process_data(baseline_folder, mitigated_folder, output_file):
    results = []

    for filename in os.listdir(mitigated_folder):
        if filename.endswith('.csv'):
            parts = filename.split('_')
            n_t_part = parts[-1].split('.')[0]
            n = int(n_t_part.split('T')[0][1:])
            t = int(n_t_part.split('T')[1])

            mitigated_file = os.path.join(mitigated_folder, filename)
            p_mitigated = calculate_performance(mitigated_file)

            baseline_file = os.path.join(baseline_folder, filename)
            if os.path.exists(baseline_file):
                p_baseline = calculate_performance(baseline_file)

                results.append({
                    'Fault_Number': n,
                    'Fault_Type': t,
                    'Fault_Name': FAULT_NAMES.get(t, f"Unknown_{t}"),
                    'P_Mitigated': p_mitigated,
                    'P_Baseline': p_baseline,
                })

    df_results = pd.DataFrame(results)
    df_results['P_Difference'] = df_results['P_Mitigated'] - df_results['P_Baseline']
    df_results = df_results.sort_values(['Fault_Type', 'Fault_Number'])
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return df_results


def plot_boxplots(df_results):
    plt.figure(figsize=(10, 5))
    
    df_melted = pd.melt(df_results, 
                        id_vars=['Fault_Number', 'Fault_Type', 'Fault_Name'], 
                        value_vars=['P_Mitigated', 'P_Baseline'], 
                        var_name='Scenario', value_name='Performance')
    
    # Map scenario names to more readable format
    df_melted['Scenario'] = df_melted['Scenario'].map({'P_Mitigated': 'MARL', 'P_Baseline': 'Baseline'})
    
    # Create the box plot with the new color palette
    sns.boxplot(x='Fault_Name', y='Performance', hue='Scenario', data=df_melted,
                width=0.15, palette=sns.color_palette("Set2")[:2])
    
    # Adjust the plot
    plt.title('Performance Comparison', fontsize=16)
    plt.ylabel('Performance', fontsize=12)
    plt.xlabel('Fault Type', fontsize=12)
    plt.xticks(rotation=0, ha='center')
    
    # Modify legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, fontsize=10, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('performance_comparison_boxplot.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("Box plot saved as performance_comparison_boxplot.png")

# def plot_boxplots(df_results):
#     plt.figure(figsize=(12, 6))
    
#     df_melted = pd.melt(df_results, 
#                         id_vars=['Fault_Number', 'Fault_Type', 'Fault_Name'], 
#                         value_vars=['P_Mitigated', 'P_Baseline'], 
#                         var_name='Condition', value_name='Performance')
    
#     # Map condition names to more readable format
#     df_melted['Condition'] = df_melted['Condition'].map({'P_Mitigated': 'MARL', 'P_Baseline': 'Baseline'})
    
#     # Create the box plot
#     sns.boxplot(x='Fault_Name', y='Performance', hue='Condition', data=df_melted,
#                 width=0.2, palette=['#1f77b4', '#ff7f0e'])
    
#     # Adjust the plot
#     plt.title('Performance Comparison: Baseline vs Mitigated', fontsize=16)
#     plt.ylabel('Performance', fontsize=12)
#     plt.xlabel('Fault Type', fontsize=12)
#     plt.xticks(rotation=0, ha='center')
#     plt.legend(title='Condition', title_fontsize='12', fontsize='10')
    
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig('performance_comparison_boxplot.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print("Box plot saved as performance_comparison_boxplot.png")

def print_summary_statistics(df_results):
    print("\nSummary Statistics:")
    summary = df_results.groupby('Fault_Name').agg({
        'P_Mitigated': ['mean', 'median', 'std'],
        'P_Baseline': ['mean', 'median', 'std'],
        'P_Difference': ['mean', 'median', 'std']
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    print(summary)
    
    # Save summary to CSV
    summary.to_csv('summary_statistics.csv')
    print("Summary statistics saved to summary_statistics.csv")

# Usage
baseline_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests/data_5/without_mitigation'
mitigated_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests/data_5/with_mitigation'
output_file = 'performance_comparison_results.csv'

df_results = process_data(baseline_folder, mitigated_folder, output_file)

# Generate visualization
plot_boxplots(df_results)

# Print and save summary statistics
print_summary_statistics(df_results)

# Print a sample of the results
print("\nSample of results:")
print(df_results.head())

# Check for performance improvements
improvements = df_results[df_results['P_Difference'] > 0].shape[0]
total = df_results.shape[0]
print(f"\nNumber of cases with performance improvement: {improvements} out of {total}")