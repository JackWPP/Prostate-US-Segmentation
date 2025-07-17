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

from .models_zoo.transunet.model import TransUNet

# --- Configuration ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train', 'masks')
TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'test', 'images')
TEST_MASK_DIR = os.path.join(DATA_DIR, 'test', 'masks')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'transunet')

# --- Hyperparameters ---
# Adjusted for a more professional training strategy for a large model
LEARNING_RATE = 1e-5  # Lower learning rate for stable convergence
BATCH_SIZE = 4      # TransUNet is memory intensive
NUM_EPOCHS = 150    # Increased epochs for proper training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIT_MODEL_NAME = 'vit_base_patch16_224_in21k'

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
        
        # Ensure mask has the correct dimensions: from (H, W, 1) to (1, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        
        if len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # Remove the last dimension: (H, W, 1) -> (H, W)
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension: (H, W) -> (1, H, W)
            
        return image, mask

# --- Loss Function ---
# Using a compound loss is generally more robust
class CompoundLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CompoundLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCELoss()  # Changed from BCEWithLogitsLoss since model outputs sigmoid

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return bce + dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred is already sigmoid activated from the model
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score

# --- Main Training Logic ---
def main():
    print(f"--- Starting TransUNet Training on {DEVICE} ---")
    print(f"Using ViT backbone: {VIT_MODEL_NAME}")
    
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    test_dataset = ProstateDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, Loss, and Scheduler
    model = TransUNet(
        img_size=256, 
        num_classes=1, 
        vit_model_name=VIT_MODEL_NAME, 
        pretrained=False  # Changed to False to avoid potential dimension issues with pretrained weights
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    criterion = CompoundLoss()

    # Training Loop
    best_dice_score = 0.0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for images, masks in progress_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Update learning rate
        scheduler.step()

        # Validation
        model.eval()
        val_dice_score = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice = (2. * intersection) / (union + 1e-6)
                val_dice_score += dice.item()

        avg_val_dice = val_dice_score / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Dice Score: {avg_val_dice:.4f}, LR: {scheduler.get_last_lr()[0]:.1e}")

        if avg_val_dice > best_dice_score:
            best_dice_score = avg_val_dice
            save_path = os.path.join(MODEL_SAVE_DIR, 'best_transunet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"-> New best model saved to {save_path} with Dice Score: {best_dice_score:.4f}")

    print("--- Training Finished ---")
    print(f"Best Validation Dice Score: {best_dice_score:.4f}")

if __name__ == "__main__":
    main()