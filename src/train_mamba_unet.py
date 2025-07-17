
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import the MambaUNet model
from src.mamba_unet import MambaUNet

# --- Configuration ---
DATA_DIR = 'processed_data/train'
MODEL_SAVE_PATH = 'models/mamba_unet_model.pth'
NUM_EPOCHS = 80
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Dataset Class ---
class ProstateDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        mask_path = os.path.join(self.data_dir, 'masks', img_name)

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Add channel dimension if missing
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask is in the correct format [C, H, W] and type
        mask = mask.permute(2, 0, 1)
        return image, mask

# --- Transformations ---
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# --- Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# --- Training Function ---
def train_model():
    print("Starting training for the Mamba-UNet model...")

    # Create dataset and dataloader
    dataset = ProstateDataset(data_dir=DATA_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Setup device, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaUNet(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_bce + loss_dice

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
