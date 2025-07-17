
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to the Python path
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models_zoo.unet.model import UNet

# --- Configuration ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train', 'masks')
TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'test', 'images')
TEST_MASK_DIR = os.path.join(DATA_DIR, 'test', 'masks')
# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50 # Adjust as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'unet')
LOG_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'training_log.csv')

# --- Dataset ---
class ProstateDataset(Dataset):
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
            
        return image, mask.unsqueeze(0) # Add channel dimension to mask

# --- Loss Function ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score

import pandas as pd

# ... (rest of the imports)

# ... (rest of the code before main)

# --- Main Training Logic ---
def main():
    print(f"--- Starting U-Net Training on {DEVICE} ---")
    
    # Data Augmentation
    train_transform = A.Compose([
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        ToTensorV2(),
    ])

    # Datasets and DataLoaders
    train_dataset = ProstateDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = ProstateDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, Loss
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DiceLoss()

    # Training Loop
    best_dice_score = 0.0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # For logging
    log_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for images, masks in progress_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_dice_score = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                
                # Calculate Dice score for validation
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice = (2. * intersection) / (union + 1e-6)
                val_dice_score += dice.item()

        avg_val_dice = val_dice_score / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Dice Score: {avg_val_dice:.4f}")
        
        # Log metrics
        log_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_dice': avg_val_dice
        })

        # Save the best model
        if avg_val_dice > best_dice_score:
            best_dice_score = avg_val_dice
            save_path = os.path.join(MODEL_SAVE_DIR, 'best_unet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"-> New best model saved to {save_path} with Dice Score: {best_dice_score:.4f}")

    # Save training log
    log_df = pd.DataFrame(log_history)
    log_save_path = os.path.join(MODEL_SAVE_DIR, 'training_log.csv')
    log_df.to_csv(log_save_path, index=False)
    print(f"--- Training Finished ---")
    print(f"Best Validation Dice Score: {best_dice_score:.4f}")
    print(f"Training log saved to {log_save_path}")

if __name__ == "__main__":
    main()
