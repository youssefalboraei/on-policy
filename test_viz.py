import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Sample data
fault_types = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
num_faults = [0, 1, 2]
controller1_performance = np.random.rand(3, 10)  # Replace with actual performance data
controller2_performance = np.random.rand(3, 10)  # Replace with actual performance data

# Calculate the difference in performance
performance_diff = controller1_performance - controller2_performance

# Create DataFrame for the difference
df_diff = pd.DataFrame(performance_diff, index=num_faults, columns=fault_types)

# Plot heatmap for performance difference
plt.figure(figsize=(12, 3))
sns.heatmap(df_diff, annot=True, cmap='coolwarm', cbar=True, xticklabels=fault_types, yticklabels=num_faults, center=0)
plt.title('Performance Difference (Controller 1 - Controller 2)')
plt.xlabel('Fault Types')
plt.ylabel('Number of Faults')
plt.show()
