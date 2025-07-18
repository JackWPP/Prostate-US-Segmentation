

import torch
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
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

# Colors for contours (BGR format for OpenCV)
COLOR_GROUND_TRUTH = (0, 255, 0)  # Green
COLOR_PREDICTION = (0, 0, 255)   # Red

# --- Model Loading ---
def load_models():
    """Load all trained models into a dictionary."""
    models = {}
    models_to_load = [
        ("MicroSegNet", MicroSegNet(num_classes=1), os.path.join(MODELS_DIR, "best_microsegnet_model.pth")),
        ("MicroSegNetAttention", MicroSegNetAttention(num_classes=1), os.path.join(MODELS_DIR, "attention", "best_attention_model.pth")),
        ("UNet", UNet(n_channels=1, n_classes=1), os.path.join(MODELS_DIR, "unet", "best_unet_model.pth"))
    ]
    for name, model_instance, path in models_to_load:
        try:
            model_instance.load_state_dict(torch.load(path, map_location=DEVICE))
            model_instance.to(DEVICE)
            model_instance.eval()
            models[name] = model_instance
            print(f"Successfully loaded {name}.")
        except Exception as e:
            print(f"Could not load {name}: {e}")
    return models

# --- Prediction Logic ---
def get_predictions(image_np, models):
    transform_unet = A.Compose([ToTensorV2()])
    predictions = {}
    with torch.no_grad():
        for name, model in models.items():
            if "MicroSegNet" in name:
                input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            elif "UNet" in name:
                augmented = transform_unet(image=image_np)
                input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
            
            output = model(input_tensor)
            
            if name == 'UNet':
                pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
            else:
                pred = (output > 0.5).cpu().numpy()
            
            predictions[name] = np.squeeze(pred).astype(np.uint8)
    return predictions

# --- Plotting Logic ---
def create_comparison_grid(image_name, models):
    """Generates and saves a 2x2 grid comparing model predictions."""
    img_path = os.path.join(TEST_IMAGE_DIR, image_name)
    mask_path = os.path.join(TEST_MASK_DIR, image_name)

    if not os.path.exists(img_path):
        print(f"Fatal: Image file not found at {img_path}")
        return

    image_np = np.load(img_path).astype(np.float32)
    mask_np = np.load(mask_path).astype(np.uint8)
    if mask_np.ndim == 3:
        mask_np = np.squeeze(mask_np, axis=-1)

    predictions = get_predictions(image_np, models)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.style.use('default')
    
    plot_titles = ["A: Ground Truth", "B: MicroSegNet", "C: MicroSegNetAttention", "D: UNet"]
    model_order = ["MicroSegNet", "MicroSegNetAttention", "UNet"]

    # Plot A: Ground Truth
    ax = axes[0, 0]
    img_gt = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours_gt, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_gt, contours_gt, -1, COLOR_GROUND_TRUTH, 2)
    ax.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    ax.text(0.05, 0.95, plot_titles[0], transform=ax.transAxes, fontsize=14, verticalalignment='top', fontweight='bold', color='white')
    ax.axis('off')

    # Plot B, C, D: Model Predictions
    for i, model_name in enumerate(model_order):
        row, col = (0, 1) if i == 0 else (1, 0) if i == 1 else (1, 1)
        ax = axes[row, col]
        
        img_pred = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if model_name in predictions:
            pred_mask = predictions[model_name]
            contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_pred, contours_pred, -1, COLOR_PREDICTION, 2)
        
        ax.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
        ax.text(0.05, 0.95, plot_titles[i+1], transform=ax.transAxes, fontsize=14, verticalalignment='top', fontweight='bold', color='white')
        ax.axis('off')

    # Create a single legend for the whole figure
    legend_elements = [Line2D([0], [0], color='g', lw=2, label='Ground Truth'),
                       Line2D([0], [0], color='r', lw=2, label='Model Prediction')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for legend
    
    # Save the plot
    output_filename = f"comparison_grid_{os.path.splitext(image_name)[0]}.png"
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"\n--- Comparison Grid Generated ---")
    print(f"Result saved to: {save_path}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 2x2 comparison grid for a specific test image.")
    parser.add_argument(
        "filename", 
        type=str, 
        help="The filename of the test image to process (e.g., 'patient_001_slice_002.npy')."
    )
    args = parser.parse_args()

    print("Loading models...")
    loaded_models = load_models()
    if not loaded_models:
        print("Fatal: No models could be loaded. Aborting.")
    else:
        create_comparison_grid(args.filename, loaded_models)

