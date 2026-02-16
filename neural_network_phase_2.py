import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from utils import create_cassis_crism_combined_mask, create_np_image
import torchvision.utils as vutils
from torchvision.transforms import v2
from scipy.ndimage import gaussian_filter


def get_crism_cassis_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=4,  # 4 channels for Cassis
        classes=3  # 3 channels for CRISM
    )

def get_transforms(patch_size=256):
    transforms = v2.Compose([
        v2.RandomApply([
            v2.RandomRotation(degrees=20, interpolation=v2.InterpolationMode.BILINEAR, fill=-10.0)
        ], p=0.5),
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


class CaSSISCRISMDataset(Dataset):
    def __init__(self, cassis_files, crism_files, transform=None):
        self.cassis_files = cassis_files
        self.crism_files = crism_files
        self.transform = transform

    def __len__(self):
        return len(self.cassis_files)
    
    def __getitem__(self, idx):
        cassis_path = self.cassis_files[idx]
        crism_path = self.crism_files[idx]

        with rasterio.open(cassis_path) as cassis_ds:
            cassis_data = cassis_ds.read().astype(np.float32)

        with rasterio.open(crism_path) as crism_ds:
            crism_data = crism_ds.read().astype(np.float32)

        cassis_data = torch.from_numpy(cassis_data)  # (4, H, W)
        crism_data = torch.from_numpy(crism_data)  # (3, H, W)
        cassis_data = torch.nan_to_num(cassis_data, nan=-10.0)
        crism_data = torch.nan_to_num(crism_data, nan=-10.0)

        if self.transform:
            for _ in range(10):
                transformed = self.transform(torch.cat([cassis_data, crism_data], dim=0))  
                cassis_data, crism_data = transformed[0:4], transformed[4:7] 
                mask = torch.tensor(create_cassis_crism_combined_mask(cassis_data.cpu().detach().numpy(), crism_data.cpu().detach().numpy(), -10.0)).unsqueeze(0)

                valid_pixel_ratio = mask.sum().item() / mask.shape[1:].numel()
                if valid_pixel_ratio == 1:
                    break
            else:
                print(f"Rejected patch from index {idx}: valid ratio = {valid_pixel_ratio:.2f}")
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx)
        else:
            mask = torch.tensor(
                create_cassis_crism_combined_mask(
                    cassis_data.cpu().detach().numpy(),
                    crism_data.cpu().detach().numpy(),
                    -10.0,
                    -10.0,
                )
            ).unsqueeze(0)

        return cassis_data, crism_data, mask
    

def train_masked_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, writer):
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

            writer.add_scalar("Loss/Train", loss.item(), epoch)

            if epoch % 10 == 0:
                detached_inputs = ctx_data.cpu().detach().numpy()
                detached_outputs = outputs.cpu().detach().numpy()
                detached_crism_data = crism_data.cpu().detach().numpy()
                detached_mask = mask.cpu().detach().numpy()
                input_img = torch.tensor(create_np_image(detached_inputs[0], detached_mask[0]))
                target_img = torch.tensor(create_np_image(detached_crism_data[0], detached_mask[0]))
                output_img = torch.tensor(create_np_image(detached_outputs[0], detached_mask[0]))
                grid = vutils.make_grid([target_img, output_img], nrow=2)
                writer.add_image("CRISM/Prediction/Train", grid, epoch)
                writer.add_image("CTX/Input/Train", input_img, epoch)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}")

        validate_masked_model(model, val_loader, criterion, device, epoch, writer)

    return model


def validate_masked_model(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for ctx_data, crism_data, mask in val_loader:
            ctx_data, crism_data, mask = ctx_data.to(device), crism_data.to(device), mask.to(device)

            outputs = model(ctx_data)
            loss = criterion(outputs, crism_data, mask)
            val_loss += loss.item()

            if writer is not None:
                if epoch % 10 == 0:
                    detached_outputs = outputs.cpu().detach().numpy()
                    detached_crism_data = crism_data.cpu().detach().numpy()
                    log_epochs(writer, epoch, detached_outputs, detached_crism_data)

                writer.add_scalar("Loss/Val", loss.item(), epoch)


class MaskedMSELoss(torch.nn.Module):
    def __init__(self, third_channel_factor: float = 1.0):
        super(MaskedMSELoss, self).__init__()
        self.third_channel_factor = third_channel_factor
    
    def forward(self, pred, target, mask):

        if pred.ndim == 3 and mask.ndim == 3:
            mask = mask.expand_as(pred)

        target = torch.nan_to_num(target, nan=-10.0)
        pred = torch.nan_to_num(pred, nan=-10.0)
        err = (pred - target) ** 2

        if err.ndim >= 3 and self.third_channel_factor != 1.0:
            channel_dim = err.ndim - 3  # channel axis for (C,H,W) or (N,C,H,W)
            if err.shape[channel_dim] >= 3:
                weight_shape = [1] * err.ndim
                weight_shape[channel_dim] = err.shape[channel_dim]
                channel_weights = torch.ones(err.shape[channel_dim], dtype=err.dtype, device=err.device)
                channel_weights[2] = self.third_channel_factor
                err = err * channel_weights.view(weight_shape)

        masked_err = err * mask

        valid_pixels = mask.sum().clamp(min=1)
        loss = masked_err.sum() / valid_pixels

        if torch.isnan(loss):
            raise ValueError("NaN detected in loss")

        return loss
    
    
class MaskedMBELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMBELoss, self).__init__()
    
    def forward(self, pred, target, mask):
        if pred.ndim == 3 and mask.ndim == 3:
            mask = mask.expand_as(pred)

        target = torch.nan_to_num(target, nan=-10.0)
        pred = torch.nan_to_num(pred, nan=-10.0)

        err = pred - target
        masked_err = err * mask

        valid_pixels = mask.sum().clamp(min=1)
        mbe = masked_err.sum() / valid_pixels

        if torch.isnan(mbe):
            raise ValueError("NaN detected in mean bias error")

        return mbe
    

class MaskedMAELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, pred, target, mask):
        if pred.ndim == 3 and mask.ndim == 3:
            mask = mask.expand_as(pred)

        target = torch.nan_to_num(target, nan=-10.0)
        pred = torch.nan_to_num(pred, nan=-10.0)

        err = torch.abs(pred - target)
        masked_err = err * mask

        valid_pixels = mask.sum().clamp(min=1)
        mae = masked_err.sum() / valid_pixels

        if torch.isnan(mae):
            raise ValueError("NaN detected in mean absolute error")

        return mae


def evaluate_crism_cassis_model(model, cassis_image, patch_size, device):
    model = model.to(device)
    model.eval()

    _, h, w = cassis_image.shape

    new_h = (h + patch_size - 1) // patch_size * patch_size  
    new_w = (w + patch_size) // patch_size * patch_size  

    pad_h = new_h - h
    pad_w = new_w - w

    cassis_padded = np.pad(cassis_image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    output_image = np.zeros((3, new_h, new_w), dtype=np.float32)


    with torch.no_grad():
        for i in range(0, new_h, patch_size):
            for j in range(0, new_w, patch_size):
                patch = cassis_padded[:, i:i + patch_size, j:j + patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                output_patch = model(patch_tensor).cpu().numpy().squeeze()
                output_image[:, i:i + patch_size, j:j + patch_size] = output_patch

    output_image = output_image[:, :h, :w]

    return output_image


def evaluate_cassis_model_with_gaussian(
    model,
    cassis_image,
    patch_size,
    stride,
    device,
    use_gaussian_weights=True,
):
    """
    Runs Gaussian-weighted tiled inference for a model trained on CASSIS input cubes.

    Args:
        model (torch.nn.Module): Trained network that maps CASSIS patches to CRISM spectra.
        cassis_image (np.ndarray): Input cube shaped (C, H, W).
        patch_size (int): Square patch size expected by the model.
        stride (int): Sliding-window stride (use patch_size for non-overlap).
        device (torch.device): CUDA or CPU device.
        use_gaussian_weights (bool): If False, simple averaging is used.

    Returns:
        np.ndarray: Reconstructed CRISM cube shaped (3, H, W).
    """
    model = model.to(device)
    model.eval()

    _, h, w = cassis_image.shape
    new_h = (h + patch_size - 1) // patch_size * patch_size
    new_w = (w + patch_size - 1) // patch_size * patch_size
    pad_h, pad_w = new_h - h, new_w - w

    cassis_padded = np.pad(cassis_image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    output_image = np.zeros((3, new_h, new_w), dtype=np.float32)
    weight_sum = np.zeros((new_h, new_w), dtype=np.float32)
    gaussian_weights = generate_gaussian_kernel(patch_size).numpy() if use_gaussian_weights else None

    with torch.no_grad():
        for i in range(0, new_h - patch_size + 1, stride):
            for j in range(0, new_w - patch_size + 1, stride):
                patch = cassis_padded[:, i:i + patch_size, j:j + patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                output_patch = model(patch_tensor).cpu().numpy().squeeze()

                if use_gaussian_weights:
                    for c in range(3):
                        output_image[c, i:i + patch_size, j:j + patch_size] += output_patch[c] * gaussian_weights
                    weight_sum[i:i + patch_size, j:j + patch_size] += gaussian_weights
                else:
                    for c in range(3):
                        output_image[c, i:i + patch_size, j:j + patch_size] += output_patch[c]
                    weight_sum[i:i + patch_size, j:j + patch_size] += 1

    weight_sum = np.where(weight_sum == 0, 1, weight_sum)
    output_image /= weight_sum
    return output_image[:, :h, :w]


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


def log_epochs(writer, epoch, outputs, crism_data):
    sample_pred = outputs[0]
    sample_target = crism_data[0]

    for i in range(3):  # CRISM has 3 channels
        pred_channel = sample_pred[i].flatten()
        target_channel = sample_target[i].flatten()

        # Histogram of predicted & target values
        # writer.add_histogram(f"CRISM_Channel_{i+1}/Prediction", pred_channel, epoch, bins='auto')
        # writer.add_histogram(f"CRISM_Channel_{i+1}/Target", target_channel, epoch, bins='auto')

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
