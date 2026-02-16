import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from utils import create_np_image
import torchvision.utils as vutils
from torchvision.transforms import v2
from scipy.ndimage import gaussian_filter

def get_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3
    )


def get_transforms(patch_size=256):
    transforms = v2.Compose([
        v2.RandomCrop(size=(patch_size, patch_size)),
        v2.Resize(size=patch_size),
        v2.ToDtype(torch.float32)
    ])
    return transforms

def get_val_transforms(patch_size=256):
    transforms = v2.Compose([
        v2.RandomCrop(size=(patch_size, patch_size)),
        v2.ToDtype(torch.float32)
    ])
    return transforms


def get_train_augmentations(patch_size=256):
    d = patch_size

    train_transform = A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(int(0.6 * d), int(0.8 * d)),
                size=(d, d),
                w2h_ratio=1.0
            ),
            ToTensorV2(),
        ],
    )

    return train_transform


def get_val_augmentations(patch_size=256):
    val_transform = A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(int(0.6 * patch_size), int(0.8 * patch_size)),
                size=(patch_size, patch_size),
                w2h_ratio=1.0
            ),
            ToTensorV2(),
        ],
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

        with rasterio.open(crism_path) as crism_ds:
            crism_data = crism_ds.read().astype(np.float32)  # Multi-channel CRISM target

        ctx_data = torch.from_numpy(ctx_data)  # (1, H, W)
        crism_data = torch.from_numpy(crism_data)  # (3, H, W)

        if self.transform:
            transformed = self.transform(torch.cat([ctx_data, crism_data], dim=0))  
            ctx_data, crism_data = transformed[0:1], transformed[1:4] 
        return ctx_data, crism_data


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, writer):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for ctx_data, crism_data in train_loader:
            ctx_data, crism_data = ctx_data.to(device), crism_data.to(device)

            optimizer.zero_grad()
            outputs = model(ctx_data)

            loss = criterion(outputs, crism_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            writer.add_scalar("Loss/Train", loss.item(), epoch)

            if epoch % 10 == 0:
                detached_inputs = ctx_data.cpu().detach().numpy()
                detached_outputs = outputs.cpu().detach().numpy()
                detached_crism_data = crism_data.cpu().detach().numpy()
                input_img = torch.tensor(create_np_image(detached_inputs[0]))
                target_img = torch.tensor(create_np_image(detached_crism_data[0]))
                output_img = torch.tensor(create_np_image(detached_outputs[0]))
                grid = vutils.make_grid([target_img, output_img], nrow=2)
                writer.add_image("CRISM/Prediction/Train", grid, epoch)
                writer.add_image("CTX/Input/Train", input_img, epoch)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}")

        validate_model(model, val_loader, criterion, device, epoch, writer)

    return model


def validate_model(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for ctx_data, crism_data in val_loader:
            ctx_data, crism_data = ctx_data.to(device), crism_data.to(device)

            outputs = model(ctx_data)
            loss = criterion(outputs, crism_data)
            val_loss += loss.item()

            if writer is not None:
                if epoch % 10 == 0:
                    detached_outputs = outputs.cpu().detach().numpy()
                    detached_crism_data = crism_data.cpu().detach().numpy()
                    log_epochs(writer, epoch, detached_outputs, detached_crism_data)

                writer.add_scalar("Loss/Val", loss.item(), epoch)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


def log_epochs(writer, epoch, outputs, crism_data):
    sample_pred = outputs[0]
    sample_target = crism_data[0]

    for i in range(3):  # CRISM has 3 channels
        pred_channel = sample_pred[i].flatten()
        target_channel = sample_target[i].flatten()

        # Histogram of predicted & target values
        writer.add_histogram(f"CRISM_Channel_{i+1}/Prediction", pred_channel, epoch, bins='auto')
        writer.add_histogram(f"CRISM_Channel_{i+1}/Target", target_channel, epoch, bins='auto')

        # Mean & Variance logging
        writer.add_scalar(f"CRISM_Channel_{i+1}/Pred_Mean", np.mean(pred_channel), epoch)
        writer.add_scalar(f"CRISM_Channel_{i+1}/Target_Mean", np.mean(target_channel), epoch)
        writer.add_scalar(f"CRISM_Channel_{i+1}/Pred_Variance", np.var(pred_channel), epoch)
        writer.add_scalar(f"CRISM_Channel_{i+1}/Target_Variance", np.var(target_channel), epoch)

        # Min & Max pixel values
        writer.add_scalar(f"CRISM_Channel_{i+1}/Pred_Min", np.min(pred_channel), epoch)
        writer.add_scalar(f"CRISM_Channel_{i+1}/Target_Min", np.min(target_channel), epoch)
        writer.add_scalar(f"CRISM_Channel_{i+1}/Pred_Max", np.max(pred_channel), epoch)
        writer.add_scalar(f"CRISM_Channel_{i+1}/Target_Max", np.max(target_channel), epoch)

        target_img = torch.tensor(create_np_image(crism_data[0]))
        output_img = torch.tensor(create_np_image(outputs[0]))
        grid = vutils.make_grid([target_img, output_img], nrow=2)
        writer.add_image("CRISM/Prediction/Val", grid, epoch)


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


def generate_gaussian_kernel(patch_size, sigma=0.5):
    """
    Generates a 2D Gaussian weight kernel for blending overlapping patches.

    Args:
        patch_size (int): Size of the square patch (H, W).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Gaussian weight matrix of shape (H, W).
    """
    kernel = np.ones((patch_size, patch_size), dtype=np.float32)
    kernel = gaussian_filter(kernel, sigma=(sigma * patch_size))
    kernel /= kernel.max()  # Normalize to 1

    return torch.tensor(kernel, dtype=torch.float32)


def evaluate_model_with_gaussian(model, ctx_image, patch_size, stride, device, use_gaussian_weights=True):
    """
    Evaluates the model on a full CTX image using a Gaussian-weighted averaging approach.

    Args:
        model (torch.nn.Module): Trained UNet model.
        ctx_image (np.ndarray): Input CTX image (H, W).
        patch_size (int): Size of square patches.
        stride (int): Step size for extracting patches.
        device (torch.device): CUDA or CPU.
        use_gaussian_weights (bool): Whether to apply Gaussian weights to the output patches.

    Returns:
        np.ndarray: Reconstructed full CRISM image.
    """
    model = model.to(device)
    model.eval()

    h, w = ctx_image.shape

    # Ensure image dimensions are divisible by patch size
    new_h = (h + patch_size - 1) // patch_size * patch_size
    new_w = (w + patch_size - 1) // patch_size * patch_size

    # Pad image with reflection to avoid boundary artifacts
    pad_h, pad_w = new_h - h, new_w - w
    ctx_padded = np.pad(ctx_image, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Initialize output arrays
    output_image = np.zeros((3, new_h, new_w), dtype=np.float32)
    weight_sum = np.zeros((new_h, new_w), dtype=np.float32)

    # Generate Gaussian weight kernel
    gaussian_weights = generate_gaussian_kernel(patch_size).numpy() if use_gaussian_weights else None

    with torch.no_grad():
        for i in range(0, new_h - patch_size + 1, stride):
            for j in range(0, new_w - patch_size + 1, stride):
                # Extract CTX patch
                patch = ctx_padded[i:i + patch_size, j:j + patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # Get model prediction
                output_patch = model(patch_tensor).cpu().numpy().squeeze()

                # Apply Gaussian-weighted summation
                if use_gaussian_weights:
                    for c in range(3):
                        output_image[c, i:i + patch_size, j:j + patch_size] += (
                            output_patch[c] * gaussian_weights
                        )
                    weight_sum[i:i + patch_size, j:j + patch_size] += gaussian_weights
                else:
                    for c in range(3):
                        output_image[c, i:i + patch_size, j:j + patch_size] += output_patch[c]
                    weight_sum[i:i + patch_size, j:j + patch_size] += 1


    # Normalize to prevent brightness inconsistencies
    weight_sum = np.where(weight_sum == 0, 1, weight_sum)  # Prevent division by zero
    output_image /= weight_sum

    return output_image[:, :h, :w]  # Crop back to original size



class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred, target, mask):
        target = torch.nan_to_num(target, nan=-10.0)
        pred = torch.nan_to_num(pred, nan=-10.0)
        err = (pred - target) ** 2
        masked_err = err * mask
        
        valid_pixels = mask.sum().clamp(min=1)
        loss = masked_err.sum() / valid_pixels

        if torch.isnan(loss):
            raise ValueError("NaN detected in loss")

        return loss
    
