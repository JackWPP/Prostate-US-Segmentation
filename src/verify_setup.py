

import torch
from torch.utils.data import DataLoader

from models_zoo.base_model.model import MicroSegNet
from train import ProstateDataset, DiceLoss

def run_verification_test():
    """
    This test verifies that the core components of the training pipeline are working correctly.
    It checks for:
    1.  Successful data loading from the preprocessed .npy files.
    2.  Correct model initialization.
    3.  A single successful forward and backward pass (one training step).
    
    If this script runs without errors, the main training script (train.py)
    is ready for execution on a server.
    """
    print("--- Starting Project Verification Test ---")
    
    # --- 1. Configuration Check ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V] Using device: {DEVICE}")
    if DEVICE == "cpu":
        print("[W] Warning: No CUDA-enabled GPU found. Test will run on CPU.")

    TRAIN_IMG_PATH = "processed_data/train/images"
    TRAIN_MASK_PATH = "processed_data/train/masks"
    
    # --- 2. Data Loading Test ---
    try:
        print("\n[T] Testing data loading...")
        test_dataset = ProstateDataset(image_dir=TRAIN_IMG_PATH, mask_dir=TRAIN_MASK_PATH)
        # Use a small batch size for a quick test
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
        images, masks = next(iter(test_loader))
        print(f"[V] Successfully loaded one batch of data.")
        print(f"[V] Image batch shape: {images.shape}")
        print(f"[V] Mask batch shape: {masks.shape}")
        assert images.shape == (2, 1, 256, 256), "Image shape is incorrect."
        assert masks.shape == (2, 1, 256, 256), "Mask shape is incorrect."
    except Exception as e:
        print(f"[E] Data loading failed: {e}")
        return

    # --- 3. Model Initialization Test ---
    try:
        print("\n[T] Testing model initialization...")
        model = MicroSegNet(num_classes=1).to(DEVICE)
        print(f"[V] Successfully initialized MicroSegNet model.")
    except Exception as e:
        print(f"[E] Model initialization failed: {e}")
        return

    # --- 4. Training Step Test (Forward + Backward Pass) ---
    try:
        print("\n[T] Testing a single training step...")
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        # Forward pass
        print("  - Performing forward pass...")
        predictions = model(images)
        print(f"[V] Forward pass successful. Prediction shape: {predictions.shape}")
        assert predictions.shape == masks.shape, "Prediction shape does not match mask shape."

        # Loss calculation
        print("  - Calculating loss...")
        loss_fn = DiceLoss()
        loss = loss_fn(predictions, masks)
        print(f"[V] Loss calculated successfully. Loss value: {loss.item():.4f}")

        # Backward pass
        print("  - Performing backward pass...")
        loss.backward()
        print(f"[V] Backward pass successful.")
        
    except Exception as e:
        print(f"[E] Training step failed: {e}")
        return

    print("\n--- Verification Complete ---")
    print("[SUCCESS] All tests passed. The project is ready for training on the server.")

if __name__ == "__main__":
    run_verification_test()

