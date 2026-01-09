import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
results_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results'
output_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/plots'
os.makedirs(output_dir, exist_ok=True)

# Define models and their files
models = {
    'LAMAR Pre-trained': ['LAMAR_layer5_perf.csv', 'LAMAR_layer11_perf.csv'],
    'LAMAR TAPT': ['LAMAR_tapt_layer5_perf.csv', 'LAMAR_tapt_layer11_perf.csv'],
    'LAMAR Random': ['LAMAR_layer5_random_perf.csv', 'LAMAR_layer11_random_perf.csv']
}

# Get unique TFs for color mapping
all_tfs = set()
for model_name, files in models.items():
    for file in files:
        path = os.path.join(results_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_tfs.update(df['TF'].unique())
all_tfs = sorted(list(all_tfs))

# Create color palette
colors = sns.color_palette('tab10', len(all_tfs))
tf_color_map = dict(zip(all_tfs, colors))

for model_name, files in models.items():
    dfs = []
    for file in files:
        path = os.path.join(results_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
    
    if not dfs:
        print(f"No data for {model_name}")
        continue
    
    # Combine data
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Group by TF and average AUROC
    avg_auroc = df_combined.groupby('TF')['AUROC'].mean().reset_index()
    
    # Overall average AUROC
    overall_avg = avg_auroc['AUROC'].mean()
    
    # Plot
    plt.figure(figsize=(12, 6))
    for tf in all_tfs:
        if tf in avg_auroc['TF'].values:
            value = avg_auroc.loc[avg_auroc['TF'] == tf, 'AUROC'].values[0]
            plt.scatter(tf, value, color=tf_color_map[tf], s=100, label=tf if tf not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(tf, 0, color=tf_color_map[tf], s=100, alpha=0.3)  # Placeholder if missing
    
    plt.xlabel('Transcription Factor (TF)')
    plt.ylabel('Average AUROC')
    plt.title(f'Average AUROC per TF for {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(title='TF', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add overall average as text
    plt.text(0.95, 0.95, f'Overall Average AUROC: {overall_avg:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_auroc_per_tf_dots.png')
    plt.savefig(save_path)
    print(f'Saved plot for {model_name} to {save_path}')
    print(f'Overall Average AUROC for {model_name}: {overall_avg:.4f}')
    plt.close()