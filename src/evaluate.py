

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from medpy.metric.binary import hd95
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.upsampling')

# --- Project Setup ---
# Add project root to the Python path for consistent imports
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import models dynamically
from src.models_zoo.unet.model import UNet
from src.models_zoo.attention_model.model import MicroSegNetAttention
from src.models_zoo.base_model.model import MicroSegNet
from src.models_zoo.transunet.model import TransUNet # Assuming this is the correct class name

# --- Configuration ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'test', 'images')
TEST_MASK_DIR = os.path.join(DATA_DIR, 'test', 'masks')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'evaluation_results.csv')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metrics Implementation ---
def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    return (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, target, smooth=1e-6):
    true_positive = np.sum(pred * target)
    predicted_positive = np.sum(pred)
    return (true_positive + smooth) / (predicted_positive + smooth)

def recall_score(pred, target, smooth=1e-6):
    true_positive = np.sum(pred * target)
    actual_positive = np.sum(target)
    return (true_positive + smooth) / (actual_positive + smooth)

def hausdorff_distance_95(pred, target):
    """
    Calculates the 95th percentile of the Hausdorff Distance using the MedPy library.
    """
    # Force squeeze to ensure 2D array
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.nan
    return hd95(pred, target)

def average_surface_distance(pred, target):
    """
    Placeholder for Average Surface Distance.
    Requires surface extraction (e.g., using marching cubes or contour finding)
    and is more complex to implement robustly from scratch.
    """
    # This is a placeholder and does not compute the actual ASD.
    return np.nan 

# --- Dataset ---
class EvaluationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask, self.images[index]

# --- Main Evaluation Logic ---
def get_model_instance(model_name):
    """Factory function to get a model instance based on its name."""
    if 'unet' in model_name and 'transunet' not in model_name:
        return UNet(n_channels=1, n_classes=1)
    elif 'attention' in model_name:
        return MicroSegNetAttention(n_channels=1, n_classes=1)
    elif 'transunet' in model_name:
        # Correct instantiation based on the project's model.py
        return TransUNet(
            img_size=256, 
            num_classes=1, 
            vit_model_name='vit_base_patch16_224_in21k', 
            pretrained=False
        )
    elif 'base' in model_name: # Assuming 'base' for MicroSegNet
        return MicroSegNet(n_channels=1, n_classes=1)
    else:
        print(f"Warning: Model name '{model_name}' not recognized. Skipping.")
        return None

def evaluate_model(model, dataloader, device):
    model.eval()
    all_metrics = {
        'dice': [], 'iou': [], 'precision': [], 'recall': [], 'hausdorff_95': []
    }

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks_np = masks.numpy()
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np.squeeze(1) # Handle (B, 1, H, W) -> (B, H, W)

            outputs = model(images)
            preds = (outputs > 0.5).cpu().numpy().squeeze(1) # (B, H, W)

            for i in range(preds.shape[0]):
                pred_mask = preds[i]
                true_mask = masks_np[i]
                
                all_metrics['dice'].append(dice_coefficient(pred_mask, true_mask))
                all_metrics['iou'].append(iou_score(pred_mask, true_mask))
                all_metrics['precision'].append(precision_score(pred_mask, true_mask))
                all_metrics['recall'].append(recall_score(pred_mask, true_mask))
                
                # Ensure masks are integer type for MedPy
                pred_mask_int = pred_mask.astype(np.uint8)
                true_mask_int = true_mask.astype(np.uint8)
                all_metrics['hausdorff_95'].append(hausdorff_distance_95(pred_mask_int, true_mask_int))

    # Calculate average of all metrics
    avg_metrics = {key: np.nanmean(values) for key, values in all_metrics.items()}
    return avg_metrics

def main():
    print(f"--- Starting Model Evaluation on {DEVICE} ---")
    
    # Prepare dataset
    eval_transform = A.Compose([ToTensorV2()])
    eval_dataset = EvaluationDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

    # --- Define Models to Evaluate ---
    models_to_evaluate = [
        {
            "name": "MicroSegNet",
            "path": os.path.join(MODELS_DIR, "best_microsegnet_model.pth"),
            "class": MicroSegNet(num_classes=1)
        },
        {
            "name": "MicroSegNetAttention",
            "path": os.path.join(MODELS_DIR, "attention", "best_microsegnet_attention_model.pth"),
            "class": MicroSegNetAttention(num_classes=1)
        },
        {
            "name": "UNet",
            "path": os.path.join(MODELS_DIR, "unet", "best_unet_model.pth"),
            "class": UNet(n_channels=1, n_classes=1)
        },
        {
            "name": "TransUNet",
            "path": os.path.join(MODELS_DIR, "transunet", "best_transunet_model.pth"),
            "class": TransUNet(img_size=256, num_classes=1, vit_model_name='vit_base_patch16_224_in21k', pretrained=False)
        }
    ]

    results = []

    for model_info in models_to_evaluate:
        model_name = model_info["name"]
        model_path = model_info["path"]
        model = model_info["class"]

        if not os.path.exists(model_path):
            print(f"No model file found for '{model_name}' at {model_path}. Skipping.")
            continue

        print(f"\n--- Evaluating: {model_name} ---")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}. Skipping.")
            continue
            
        avg_metrics = evaluate_model(model, eval_loader, DEVICE)
        avg_metrics['model'] = model_name
        results.append(avg_metrics)

        print(f"Results for {model_name}:")
        for key, value in avg_metrics.items():
            if key != 'model':
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df = df[['model', 'dice', 'iou', 'precision', 'recall', 'hausdorff_95']] # Reorder columns
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n--- Evaluation complete. Results saved to {OUTPUT_FILE} ---")
        print(df.to_string())
    else:
        print("\n--- No models were evaluated. ---")

if __name__ == "__main__":
    main()

