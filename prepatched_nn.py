import rasterio
from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torchvision.utils as vutils


class PatchedDataset(Dataset):
    def __init__(self, ctx_files, crism_files):
        self.ctx_files = ctx_files
        self.crism_files = crism_files

    def __len__(self):
        return len(self.ctx_files)
    
    def __getitem__(self, idx):
        ctx_path = self.ctx_files[idx]
        crism_path = self.crism_files[idx]

        ctx_data = torch.load(ctx_path)
        crism_data = torch.load(crism_path)

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
            # if epoch % 10 == 0:
            #     writer.add_histogram("CRISM/Target Distribution", crism_data, epoch)
            #     writer.add_histogram("Output/Prediction Distribution", outputs, epoch)
                

                # writer.add_image("CTX/Input", create_image(ctx_data[0]), epoch)
                # writer.add_image("CRISM/Target", create_image(crism_data[0]), epoch)
                # writer.add_image("Output/Prediction", create_image(outputs[0]), epoch)

            if epoch % 10 == 0:
                target_img = torch.tensor(create_np_image(crism_data[0]))
                output_img = torch.tensor(create_np_image(outputs[0]))
                grid = vutils.make_grid([target_img, output_img], nrow=2)
                writer.add_image("CRISM/Prediction/Val", grid, epoch)

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


def extract_valid_patches(ctx_path, crism_path, output_dir, stride=128):
    """ Extracts valid patches from CTX and CRISM while maximizing area coverage. """
    
    with rasterio.open(ctx_path) as ctx_ds, rasterio.open(crism_path) as crism_ds:
        ctx_data = ctx_ds.read().astype(np.float32)
        crism_data = crism_ds.read().astype(np.float32)

        # Generate mask (valid pixels are present in both images)
        mask = create_mask(crism_data, 65535)

        _, h, w = ctx_data.shape

        patch_id = 0
        for i in range(0, h - 256, stride):
            for j in range(0, w - 256, stride):
                ctx_patch = ctx_data[:, i: i + 256, j: j + 256]
                crism_patch = crism_data[:, i: i + 256, j: j + 256]

                # Ensure patch is valid
                mask_patch = mask[i: i + 256, j: j + 256]
                if np.any(mask_patch == 0):  # Skip if any invalid pixels exist
                    continue  

                # Save patches
                filename = split_path_into_dir_and_file(ctx_path)[1].split(".")[0]
                torch.save(ctx_patch, os.path.join(output_dir, f"ctx_{filename}_{patch_id}.pt"))
                torch.save(crism_patch, os.path.join(output_dir, f"crism_{filename}_{patch_id}.pt"))
                
                patch_id += 1
        print(f"Extracted {patch_id} patches from {ctx_path}")


def split_into_train_test_val(ctx_dir, crism_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """ Splits CTX and CRISM files into train, validation, and test sets and saves them into corresponding dirs. """
    ctx_files = get_files_from_dir(ctx_dir, ".pt")
    crism_files = get_files_from_dir(crism_dir, ".pt")

    assert len(ctx_files) == len(crism_files), "Number of CTX and CRISM files must match."

    num_files = len(ctx_files)
    indices = np.arange(num_files)
    np.random.shuffle(indices)

    train_end = int(num_files * train_ratio)
    val_end = int(num_files * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    for split, indices in zip(["train", "val", "test"], [train_indices, val_indices, test_indices]):
        ctx_split_dir = os.path.join(output_dir, "ctx", split)
        crism_split_dir = os.path.join(output_dir, "crism", split)

        os.makedirs(ctx_split_dir, exist_ok=True)
        os.makedirs(crism_split_dir, exist_ok=True)

        for idx in indices:
            ctx_file = ctx_files[idx]
            crism_file = crism_files[idx]

            ctx_file_name = os.path.basename(ctx_file)
            crism_file_name = os.path.basename(crism_file)

            torch.save(torch.load(ctx_file), os.path.join(ctx_split_dir, ctx_file_name))
            torch.save(torch.load(crism_file), os.path.join(crism_split_dir, crism_file_name))
        print(f"Saved {split} split to {output_dir}")