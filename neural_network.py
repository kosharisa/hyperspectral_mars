import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import rasterio
from utils import create_mask

def get_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3
    )


def get_train_augmentations(patch_size=256):
    d = patch_size

    train_transform = A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(int(0.6 * d), int(0.8 * d)),
                size=(d, d),
                w2h_ratio=1.0
            ),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.25, fill_mask=65535, fill=0),
            A.CoarseDropout(
                hole_height_range=(int(0.2 * d), int(0.4 * d)),
                hole_width_range=(int(0.2 * d), int(0.4 * d)),
                fill_mask=65535,
                fill=0,
                p=0.25
            ),
            ToTensorV2(),
        ]
    )

    return train_transform


def get_val_augmentations(patch_size=256):
    val_transform = A.Compose(
        [
            A.CenterCrop(height=patch_size, width=patch_size),
            ToTensorV2(),
        ]
    )

    return val_transform


class CTXCRISMDataset(Dataset):
    def __init__(self, ctx_files, crism_files, transform=None):
        self.ctx_files = ctx_files
        self.crism_files = crism_files
        self.transform = transform

    def __len__(self):
        return len(self.ctx_files)
    
    def __getitem__(self, idx):
        ctx_path = self.ctx_files[idx]
        crism_path = self.crism_files[idx]

        with rasterio.open(ctx_path) as ctx_ds:
            ctx_data = ctx_ds.read().astype(np.float32)
            ctx_data = np.transpose(ctx_data, (1, 2, 0)) # Reorder to have (H, W, C) for albumentations

        with rasterio.open(crism_path) as crism_ds:
            crism_data = crism_ds.read().astype(np.float32)  # Multi-channel CRISM target
            crism_data = np.transpose(crism_data, (1, 2, 0)) # Reorder to have (H, W, C)

        if self.transform:
            augmented = self.transform(image=ctx_data, mask=crism_data)
            ctx_data, crism_data = augmented["image"], augmented["mask"]
            crism_data = crism_data.permute(2, 0, 1) # Manually reorder (C, H, W) because it is (H, W, C) after albumentations
            mask = torch.tensor(create_mask(crism_data.numpy())).unsqueeze(0) # Add C dimension to match shapes
        return ctx_data, crism_data, mask
    

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for ctx_data, crism_data, mask in train_loader:
            ctx_data, crism_data, mask = ctx_data.to(device), crism_data.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs = model(ctx_data)
            loss = criterion(outputs, crism_data, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}")

        validate_model(model, val_loader, criterion, device)

    return model


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for ctx_data, crism_data, mask in val_loader:
            ctx_data, crism_data, mask = ctx_data.to(device), crism_data.to(device), mask.to(device)

            outputs = model(ctx_data)
            loss = criterion(outputs, crism_data, mask)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


def evaluate_model(model, ctx_image, patch_size, device):
    model = model.to(device)
    model.eval()

    h, w = ctx_image.shape

    new_h = (h + patch_size - 1) // patch_size * patch_size  
    new_w = (w + patch_size) // patch_size * patch_size  

    pad_h = new_h - h
    pad_w = new_w - w

    ctx_padded = np.pad(ctx_image, ((0, pad_h), (0, pad_w)), mode='reflect')

    output_image = np.zeros((3, new_h, new_w), dtype=np.float32)


    with torch.no_grad():
        for i in range(0, new_h, patch_size):
            for j in range(0, new_w, patch_size):
                patch = ctx_padded[i:i + patch_size, j:j + patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                output_patch = model(patch_tensor).cpu().numpy().squeeze()
                output_image[:, i:i + patch_size, j:j + patch_size] = output_patch

    output_image = output_image[:, :h, :w]

    return output_image


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred, target, mask):
        err = (pred - target) ** 2
        masked_err = err * mask
        
        valid_pixels = mask.sum().clamp(min=1)

        return masked_err.sum() / valid_pixels
    
