

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- Configuration ---
MODELS_DIR = "models"
OUTPUT_DIR = "output"
sns.set_theme(style="whitegrid")

def find_log_files(models_root_dir):
    """Finds all training_log.csv files in the models directory."""
    log_files = []
    # Check for logs in subdirectories (unet, transunet, attention)
    for model_dir in os.listdir(models_root_dir):
        potential_log_path = os.path.join(models_root_dir, model_dir, "training_log.csv")
        if os.path.isfile(potential_log_path):
            log_files.append((model_dir, potential_log_path))
            
    # Check for the base microsegnet log in the root models folder
    base_log_path = os.path.join(models_root_dir, "base", "training_log.csv")
    if os.path.isfile(base_log_path):
        log_files.append(("MicroSegNet", base_log_path))
        
    return log_files

def plot_training_history(model_name, log_path, save_dir):
    """Reads a log file and plots its training loss and validation Dice score on a single chart with dual axes."""
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Log file not found for {model_name} at {log_path}. Skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Training Loss on the primary y-axis (left)
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    sns.lineplot(data=df, x='epoch', y='train_loss', marker='o', ax=ax1, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the Validation Dice Score (right)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Dice Score', color=color)
    sns.lineplot(data=df, x='epoch', y='val_dice', marker='o', ax=ax2, color=color, label='Validation Dice')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0) # Dice score is between 0 and 1

    # Title and Legend
    plt.title(f'Training History for {model_name.capitalize()}', fontsize=16)
    fig.tight_layout()
    
    # Create a single legend for both lines
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    ax1.get_legend().remove()


    # Save the plot
    save_path = os.path.join(save_dir, f"{model_name}_training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined plot for {model_name} to {save_path}")

def main():
    """Main function to find logs and generate all plots."""
    print("--- Generating Training History Plots ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_files = find_log_files(MODELS_DIR)
    
    if not log_files:
        print("No training_log.csv files found. Please run training scripts first.")
        return
        
    for model_name, log_path in log_files:
        # Use a more descriptive name for the base model plot
        if model_name == 'base':
            plot_name = 'MicroSegNet'
        else:
            plot_name = model_name
        plot_training_history(plot_name, log_path, OUTPUT_DIR)
        
    print("\n--- All plots generated. ---")

if __name__ == "__main__":
    main()

