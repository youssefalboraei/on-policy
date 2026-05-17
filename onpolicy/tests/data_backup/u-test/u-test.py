import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

FAULT_TYPES = {
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
    return df['Delivery_Rate'].iloc[-1]  # Return the last Delivery_Rate value

def process_data(base_folder, scenario):
    results = {}
    
    mitigated_folder = os.path.join(base_folder, scenario, 'with_mitigation')
    baseline_folder = os.path.join(base_folder, scenario, 'without_mitigation')
    
    for folder, condition in [(mitigated_folder, 'mitigated'), (baseline_folder, 'baseline')]:
        for file in os.listdir(folder):
            if file.startswith('simulation_data_run'):
                file_path = os.path.join(folder, file)
                fault_number = int(file.split('_N')[1].split('T')[0])
                fault_type = int(file.split('_N')[1].split('T')[1].split('.')[0])
                performance = calculate_performance(file_path)
                
                key = (fault_number, fault_type)
                if key not in results:
                    results[key] = {'mitigated': [], 'baseline': []}
                results[key][condition].append(performance)
    
    return results

def calculate_mitigation_power(D_mit, D_none):
    U, _ = stats.mannwhitneyu(D_mit, D_none, alternative='two-sided')
    
    n1, n2 = len(D_mit), len(D_none)
    
    combined = np.concatenate([D_mit, D_none])
    ranks = stats.rankdata(combined)
    R_mit = np.sum(ranks[:n1])
    R_none = np.sum(ranks[n1:])
    
    if R_mit > R_none:
        h = 1
    elif R_mit == R_none:
        h = 0
    else:
        h = -1
    
    mitigation_power = 2 *h * abs( U / (n1 * n2) - 0.5)
    return mitigation_power

def analyze_fault_mitigation(data):
    results = []
    for (fault_number, fault_type), scores in data.items():
        if scores['mitigated'] and scores['baseline']:
            mitigation_power = calculate_mitigation_power(scores['mitigated'], scores['baseline'])
            results.append({
                'Fault_Number': fault_number,
                'Fault_Type': fault_type,
                'Fault_Name': FAULT_TYPES.get(fault_type, "DEFAULT"),
                'Mitigation_Power': mitigation_power
            })
    return pd.DataFrame(results)

def prepare_heatmap_data(results):
    heatmap_data = results.pivot(index='Fault_Number', columns='Fault_Name', values='Mitigation_Power')
    
    default_zero = heatmap_data.loc[0, 'DEFAULT'] if 'DEFAULT' in heatmap_data.columns else None
    
    if default_zero is not None:
        heatmap_data.loc[0] = heatmap_data.loc[0].fillna(default_zero)
    
    heatmap_data = heatmap_data.drop('DEFAULT', axis=1, errors='ignore')
    
    return heatmap_data

def visualize_heatmap(heatmap_data, scenario):
    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, vmin=-1, vmax=1, fmt='.2f',
                annot_kws={'size': 18})
    plt.title(f'Mitigation Power Heatmap ({scenario.upper()} Scenario)', fontsize=22)
    plt.ylabel('Number of Faults', fontsize=18)
    plt.xlabel('Fault Type', fontsize=18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()
    plt.savefig(f'{scenario.lower()}_mitigation_power_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {scenario.lower()}_mitigation_power_heatmap.png")

def main(scenario):
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    data = process_data(base_folder, scenario)

    results = analyze_fault_mitigation(data)
    print(f"\nAnalysis results for {scenario.upper()} scenario:")
    print(results)

    results.to_csv(f'{scenario.lower()}_mitigation_power_analysis.csv', index=False)
    print(f"Results saved to {scenario.lower()}_mitigation_power_analysis.csv")

    heatmap_data = prepare_heatmap_data(results)
    visualize_heatmap(heatmap_data, scenario)

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