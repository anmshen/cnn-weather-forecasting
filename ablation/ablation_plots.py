import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("ablation_results.csv")

# 1. Create a fixed color mapping using the 'viridis' colormap
# This ensures each label gets a unique shade from the spectrum
unique_labels = df['label'].unique()
colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_labels)))
color_map = dict(zip(unique_labels, colors))

metrics = {
    'rmse_TMP@2m_above_ground': 'Temperature RMSE (2m)',
    'rmse_RH@2m_above_ground': 'Relative Humidity RMSE (2m)',
    'rmse_UGRD@10m_above_ground': 'U-Wind RMSE (10m)',
    'rmse_VGRD@10m_above_ground': 'V-Wind RMSE (10m)',
    'rmse_GUST@surface': 'Wind Gust RMSE',
    'rmse_APCP_1hr_acc_fcst@surface': 'Precipitation RMSE (1hr)',
    'rmse_APCP_heavy': 'Heavy Precipitation RMSE',
    'auc_precip_binary': 'Precipitation Classification AUC'
}

for metric, title in metrics.items():
    # Sort for the specific metric
    ascending = False if 'auc' in metric else True
    df_sorted = df.sort_values(by=metric, ascending=ascending).reset_index()

    plt.figure(figsize=(10, 6))

    # Apply the consistent color based on the label mapping
    bar_colors = [color_map[label] for label in df_sorted['label']]
    plt.bar(range(len(df_sorted)), df_sorted[metric], color=bar_colors)

    # 2. Angled x-axis labels (45 degrees) for better readability
    plt.xticks(
        range(len(df_sorted)), 
        df_sorted['label'], 
        rotation=45, 
        ha='right'
    )

    plt.title(title, fontweight='bold')
    plt.ylabel(title)

    # 3. Build the fixed legend using proxy artists
    # This keeps the legend order and colors identical across all output images
    handles = [plt.Rectangle((0,0), 1, 1, color=color_map[label]) for label in unique_labels]
    plt.legend(handles, unique_labels, title="Configurations", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{metric}.png")
    plt.close() # Recommended to free up memory when generating multiple plots