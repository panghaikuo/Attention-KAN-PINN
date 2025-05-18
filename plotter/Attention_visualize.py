import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.backends.backend_pdf import PdfPages

# Try to import scienceplots if available
try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except ImportError:
    print("scienceplots package not found, using default style")

# Load the NumPy file data
file_path = '../My-MIT_Attention results/Attention weight/Experiment1/attention_data/epoch_1/attention_layer_1_data.npy'
with open(file_path, 'rb') as f:
    data = np.load(f, allow_pickle=True).item()

# Extract the data
feature_before = data['feature_before']
feature_after = data['feature_after']
attention_weights = data['attention_weights']

# Create a figure with high DPI
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Set up x-axis values (feature dimensions)
feature_dimensions = np.arange(len(feature_before))

# Plot the data
ax.plot(feature_dimensions, feature_before, color='#4682B4', label='Before Attention', linewidth=2.5)
ax.plot(feature_dimensions, feature_after, color='#FF8C00', label='After Attention', linewidth=2.5)

# Plot attention weights as bars
ax.bar(feature_dimensions, attention_weights, alpha=0.5, color='#FF9999', label='Attention Weights')

# Add labels and title with bold fonts
ax.set_xlabel('Feature Dimension', fontsize=16, fontweight='bold')
ax.set_ylabel('Feature Value', fontsize=16, fontweight='bold')
ax.set_title('Layer 1: Feature Changes Before and After Attention', fontsize=18, fontweight='bold')

# Add grid
ax.grid(True, linestyle='--', alpha=0.6)

# Add legend with proper formatting - make sure labels are bold
legend = ax.legend(fontsize=14)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set y-axis limit to match the reference image
ax.set_ylim(0, 1.0)

# Set tick parameters to match the reference plot
ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=7, direction='inout',
               grid_color='black', grid_alpha=0.5)

# Thicken plot borders
plt.setp(ax.spines.values(), linewidth=2.5)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = 'attention_layer_1_visualization.png'
plt.savefig(output_path, dpi=300)
print(f"Figure saved to {output_path}")

# Display the figure
plt.show()