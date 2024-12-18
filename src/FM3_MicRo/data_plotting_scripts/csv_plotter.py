import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
# Replace 'data.csv' with the path to your CSV file
file_path = 'src/FM3_MicRo/control_models/2024-12-18_17-11-50_delta_r_10000-steps_10-obs/log.csv'
data = pd.read_csv(file_path)

# Extract the Episode and Ratio columns
episodes = data['Episode']
ratios = data['Ratio']

# Ensure that Ratio column is numeric (convert if necessary)
ratios = pd.to_numeric(ratios, errors='coerce')

# Drop rows with NaN values in Episode or Ratio
data_cleaned = data.dropna(subset=['Episode', 'Ratio'])
episodes = data_cleaned['Episode']
ratios = data_cleaned['Ratio']

# Calculate the trendline (linear fit)
z = np.polyfit(episodes, ratios, 1)  # Degree 1 polynomial (linear)
p = np.poly1d(z)

# Generate trendline values
trendline = p(episodes)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(episodes, ratios, label='Data Points', color='blue', alpha=0.7)
plt.plot(episodes, trendline, label='Trend Line', color='black')

# Add labels, title, and legend
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Ratio', fontsize=12)
plt.title('Episode vs. Ratio with Trend Line', fontsize=14)
plt.legend()

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
