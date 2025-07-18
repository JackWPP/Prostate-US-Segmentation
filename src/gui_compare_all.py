
import gradio as gr
import torch
import numpy as np
import os
import glob
from PIL import Image

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

# --- Model Loading ---
def load_models():
    """Load all trained models into a dictionary."""
    models = {}
    
    # MicroSegNet (Base)
    try:
        model_base = MicroSegNet(num_classes=1)
        path_base = os.path.join(MODELS_DIR, "best_microsegnet_model.pth")
        model_base.load_state_dict(torch.load(path_base, map_location=DEVICE))
        model_base.to(DEVICE)
        model_base.eval()
        models['MicroSegNet'] = model_base
        print("Successfully loaded MicroSegNet.")
    except Exception as e:
        print(f"Could not load MicroSegNet: {e}")

    # MicroSegNetAttention
    try:
        model_attn = MicroSegNetAttention(num_classes=1)
        path_attn = os.path.join(MODELS_DIR, "attention", "best_microsegnet_attention_model.pth")
        model_attn.load_state_dict(torch.load(path_attn, map_location=DEVICE))
        model_attn.to(DEVICE)
        model_attn.eval()
        models['MicroSegNetAttention'] = model_attn
        print("Successfully loaded MicroSegNetAttention.")
    except Exception as e:
        print(f"Could not load MicroSegNetAttention: {e}")

    # UNet
    try:
        model_unet = UNet(n_channels=1, n_classes=1)
        path_unet = os.path.join(MODELS_DIR, "unet", "best_unet_model.pth")
        model_unet.load_state_dict(torch.load(path_unet, map_location=DEVICE))
        model_unet.to(DEVICE)
        model_unet.eval()
        models['UNet'] = model_unet
        print("Successfully loaded UNet.")
    except Exception as e:
        print(f"Could not load UNet: {e}")
        
    return models

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ... (imports)

# --- Preprocessing Functions ---
# Define separate preprocessing pipelines to match each model's training
transform_microsegnet = A.Compose([
    # MicroSegNet's train.py uses permute, which is not standard.
    # We will handle this manually.
])

transform_unet = A.Compose([
    ToTensorV2(),
])

# --- Prediction Function ---
def predict(image_name, models):
    """Load an image and run predictions with all loaded models."""
    
    # Load original image and ground truth mask
    img_path = os.path.join(TEST_IMAGE_DIR, image_name)
    mask_path = os.path.join(TEST_MASK_DIR, image_name)
    
    image_np = np.load(img_path).astype(np.float32)
    mask_np = np.load(mask_path).astype(np.uint8)
    if mask_np.ndim == 3:
        mask_np = np.squeeze(mask_np, axis=-1)

    results = []
    results.append((image_np, "Original Image"))
    results.append((mask_np * 255, "Ground Truth"))

    # Generate predictions for each model
    with torch.no_grad():
        for name, model in models.items():
            
            # --- Apply Model-Specific Preprocessing ---
            if "MicroSegNet" in name:
                # Manual permute to match train.py
                input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            elif "UNet" in name:
                # Use Albumentations pipeline to match train_unet.py
                augmented = transform_unet(image=image_np)
                input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
            else:
                # Default to MicroSegNet's style if unsure
                input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # --- Get Model Output ---
            output = model(input_tensor)
            
            # Process output (handle both raw logits and sigmoid outputs)
            if name == 'UNet': # UNet model file returns raw logits
                 pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
            else: # MicroSegNet models return sigmoid output
                 pred = (output > 0.5).cpu().numpy()

            pred_mask = np.squeeze(pred).astype(np.uint8) * 255
            results.append((pred_mask, name))
            
    return results

# --- Gradio Interface ---
def build_gui():
    print("Loading models...")
    models = load_models()
    
    if not models:
        print("No models could be loaded. Aborting GUI launch.")
        return

    # Get list of test images
    test_images = sorted([os.path.basename(p) for p in glob.glob(os.path.join(TEST_IMAGE_DIR, "*.npy"))])

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Comprehensive Model Comparison for Prostate Segmentation")
        gr.Markdown("Select a test image from the dropdown to see the segmentation results from all available models.")
        
        with gr.Row():
            image_dropdown = gr.Dropdown(choices=test_images, label="Select Test Image", value=test_images[0])
        
        gallery = gr.Gallery(label="Comparison Results", show_label=True, elem_id="gallery", columns=len(models) + 2, rows=1, object_fit="contain", height="auto")

        # Link dropdown to the prediction function
        image_dropdown.change(
            fn=lambda x: predict(x, models),
            inputs=image_dropdown,
            outputs=gallery
        )
        
        # Initial load
        demo.load(
            fn=lambda x: predict(x, models),
            inputs=image_dropdown,
            outputs=gallery
        )

    print("Launching Gradio GUI...")
    demo.launch(share=True)

if __name__ == "__main__":
    build_gui()
