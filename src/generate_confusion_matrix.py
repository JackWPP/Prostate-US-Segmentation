import torch
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Project Setup ---
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import models
from src.models_zoo.base_model.model import MicroSegNet
from src.models_zoo.attention_model.model import MicroSegNetAttention
from src.models_zoo.unet.model import UNet

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test', 'images')
TEST_MASK_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test', 'masks')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# --- Model Loading & Config ---
MODELS_CONFIG = {
    "MicroSegNet": {
        "class": MicroSegNet(num_classes=1),
        "path": os.path.join(MODELS_DIR, "best_microsegnet_model.pth")
    },
    "MicroSegNetAttention": {
        "class": MicroSegNetAttention(num_classes=1),
        "path": os.path.join(MODELS_DIR, "attention", "best_attention_model.pth")
    },
    "UNet": {
        "class": UNet(n_channels=1, n_classes=1),
        "path": os.path.join(MODELS_DIR, "unet", "best_unet_model.pth")
    }
}

# --- Confusion Matrix Calculation ---
def calculate_confusion_matrix(model, model_name):
    """Calculates the pixel-wise confusion matrix over the entire test set."""
    print(f"Calculating confusion matrix for {model_name}...")
    
    test_images = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*.npy")))
    transform_unet = A.Compose([ToTensorV2()])
    
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    with torch.no_grad():
        for img_path in tqdm(test_images, desc=f"Processing for {model_name}", leave=False):
            image_name = os.path.basename(img_path)
            mask_path = os.path.join(TEST_MASK_DIR, image_name)

            image_np = np.load(img_path).astype(np.float32)
            mask_np = np.load(mask_path).astype(np.uint8)
            if mask_np.ndim == 3:
                mask_np = np.squeeze(mask_np, axis=-1)

            # Apply model-specific preprocessing
            if "MicroSegNet" in model_name:
                input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            elif "UNet" in model_name:
                augmented = transform_unet(image=image_np)
                input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
            
            output = model(input_tensor)
            
            if model_name == 'UNet':
                pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
            else:
                pred = (output > 0.5).cpu().numpy()
            
            pred_mask = np.squeeze(pred).astype(np.uint8)

            # Flatten and compare
            pred_flat = pred_mask.flatten()
            true_flat = mask_np.flatten()

            total_tp += np.sum(np.logical_and(pred_flat == 1, true_flat == 1))
            total_tn += np.sum(np.logical_and(pred_flat == 0, true_flat == 0))
            total_fp += np.sum(np.logical_and(pred_flat == 1, true_flat == 0))
            total_fn += np.sum(np.logical_and(pred_flat == 0, true_flat == 1))

    return np.array([[total_tn, total_fp], [total_fn, total_tp]])

# --- Plotting Logic ---
def plot_confusion_matrix(conf_matrix, model_name):
    """Generates and saves a final, enhanced heatmap with a color bar and percentages."""
    print(f"Generating final plot for {model_name}...")
    
    # Calculate percentages for display
    cm_sum = np.sum(conf_matrix)
    cm_perc = conf_matrix / cm_sum * 100
    
    # Create combined labels (count + percentage)
    labels = (np.asarray(["{0:d}\n({1:.2f}%)".format(value, percentage)
                         for value, percentage in zip(conf_matrix.flatten(), cm_perc.flatten())])
              ).reshape(2, 2)

    # Create a DataFrame for better labeling
    categories = ['Background', 'Prostate']
    df_cm = pd.DataFrame(conf_matrix, index=categories, columns=categories)

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.4)
    
    # Create the heatmap with the color bar enabled
    heatmap = sns.heatmap(df_cm, annot=labels, fmt='', cmap='YlGnBu', cbar=True, cbar_kws={'label': 'Pixel Count'})
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center')
    
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.title(f'Pixel-wise Confusion Matrix: {model_name}', fontsize=18, pad=20)
    
    output_filename = f"confusion_matrix_{model_name}.png"
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Result saved to: {save_path}")


# --- Main Execution ---
def main():
    """Main function to generate confusion matrices for all models."""
    print("--- Generating Confusion Matrices for All Models ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for model_name, config in MODELS_CONFIG.items():
        model = config["class"]
        path = config["path"]
        
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print(f"\nSuccessfully loaded {model_name}.")
            
            cm = calculate_confusion_matrix(model, model_name)
            plot_confusion_matrix(cm, model_name)
            
        except Exception as e:
            print(f"\nCould not process {model_name}. Error: {e}")

    print("\n--- All tasks complete. ---")

if __name__ == "__main__":
    main()