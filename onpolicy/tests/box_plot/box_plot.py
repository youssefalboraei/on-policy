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

def plot_boxplots(df_results, scenario):
    plt.figure(figsize=(15, 8))
    
    df_melted = pd.melt(df_results, 
                        id_vars=['Fault_Number', 'Fault_Type', 'Fault_Name'], 
                        value_vars=['P_Mitigated', 'P_Baseline'], 
                        var_name='Scenario', value_name='Performance')
    
    df_melted['Scenario'] = df_melted['Scenario'].map({'P_Mitigated': 'MARL', 'P_Baseline': 'Baseline'})
    
    sns.boxplot(x='Fault_Name', y='Performance', hue='Scenario', data=df_melted,
                width=0.3, palette=sns.color_palette("Set2")[:2])
    
    plt.title(f'Performance Comparison ({scenario})', fontsize=22)
    plt.ylabel('Performance', fontsize=18)
    plt.xlabel('Fault Type', fontsize=18)
    plt.xticks(rotation=0, ha='right', fontsize=14)
    plt.yticks(fontsize = 14)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{scenario.lower()}_performance_comparison_boxplot.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved as {scenario.lower()}_performance_comparison_boxplot.png")

def print_summary_statistics(df_results, scenario):
    print(f"\nSummary Statistics for {scenario}:")
    summary = df_results.groupby('Fault_Name').agg({
        'P_Mitigated': ['mean', 'median', 'std'],
        'P_Baseline': ['mean', 'median', 'std'],
        'P_Difference': ['mean', 'median', 'std']
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    print(summary)
    
    summary.to_csv(f'{scenario.lower()}_summary_statistics.csv')
    print(f"Summary statistics saved to {scenario.lower()}_summary_statistics.csv")

def main(scenario):
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    baseline_folder = os.path.join(base_folder, scenario, 'without_mitigation')
    mitigated_folder = os.path.join(base_folder, scenario, 'with_mitigation')
    output_file = f'{scenario.lower()}_performance_comparison_results.csv'

    df_results = process_data(baseline_folder, mitigated_folder, output_file)

    plot_boxplots(df_results, scenario)
    print_summary_statistics(df_results, scenario)

    print("\nSample of results:")
    print(df_results.head())

    improvements = df_results[df_results['P_Difference'] > 0].shape[0]
    total = df_results.shape[0]
    print(f"\nNumber of cases with performance improvement: {improvements} out of {total}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <scenario>")
        print("Available scenarios: RT-RE, RT-SE, ST-RE, ST-SE")
        sys.exit(1)
    
    scenario = sys.argv[1].upper()
    if scenario not in ["RT-RE", "RT-SE", "ST-RE", "ST-SE"]:
        print("Invalid scenario. Please choose from: RT-RE, RT-SE, ST-RE, ST-SE")
        sys.exit(1)
    
    main(scenario)
