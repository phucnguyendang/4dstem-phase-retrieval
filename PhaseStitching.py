import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import ViTModel, ViTConfig
import h5py
import numpy as np
import argparse
import os
from typing import Tuple, List
import math
import wandb

class PhaseStitchingDataset(Dataset):
    """Dataset for phase stitching training"""
    
    def __init__(self, data_files: List[str], stride: int = 9):
        self.data_files = data_files
        self.stride = stride
        self.samples = []
        for file_path in data_files:
            with h5py.File(file_path, 'r') as f:
                # Lấy tất cả các key dạng dp_set_{idx}
                dp_keys = [k for k in f.keys() if k.startswith("dp_set_")]
                phase_keys = set(k for k in f.keys() if k.startswith("phase_"))
                for dp_key in dp_keys:
                    idx = int(dp_key.replace("dp_set_", ""))
                    phase_key = f"phase_{idx}"
                    if phase_key in phase_keys:
                        self.samples.append((file_path, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, sample_id = self.samples[idx]
        
        with h5py.File(file_path, 'r') as f:
            # Load phase data
            phase_data = f[f"phase_{sample_id}"][:]  # Shape: (256, 256)
        
        # Extract multiple phase patches with stride=9
        phase_patches = []
        # coordinates = []
        
        # For 50x50 grid with 14x14 patches and stride=9, we get 5x5 = 25 patches
        for patch_row in range(5):
            for patch_col in range(5):
                start_row = patch_row * self.stride
                start_col = patch_col * self.stride
                
                # Calculate bright center
                bright_center_row = round((start_row*2 + 13) * 2.56)
                bright_center_col = round((start_col*2 + 13) * 2.56)
                
                # Calculate shift for ground truth phase
                probe_gpts_tuple = (256, 256)
                shift_y = probe_gpts_tuple[0] // 2 - bright_center_row
                shift_x = probe_gpts_tuple[1] // 2 - bright_center_col
                shift = (shift_y, shift_x)
                
                # Roll phase and extract 76x76 central region
                rolled_phase = np.roll(phase_data, shift, axis=(0, 1))
                center_y, center_x = rolled_phase.shape[0] // 2, rolled_phase.shape[1] // 2
                phase_patch = rolled_phase[center_y-38:center_y+38, center_x-38:center_x+38]
                
                phase_patches.append(phase_patch)
                # coordinates.append([start_row/50, start_col/50]) # Normalize to [0, 1] range, 50 is the grid size
        
        # Convert to tensors
        phase_patches = torch.FloatTensor(np.array(phase_patches))  # (25, 76, 76)
        # coordinates = torch.FloatTensor(np.array(coordinates))      # (25, 2)
        gt_full_phase = torch.FloatTensor(phase_data)               # (256, 256)
        
        # return phase_patches, coordinates, gt_full_phase
        return phase_patches, gt_full_phase

class PatchEncoder(nn.Module):
    """Encode phase patches to feature vectors"""
    
    def __init__(self, patch_size=76, embed_dim=768):
        super().__init__()
        
        # CNN encoder for phase patches
        self.patch_encoder = nn.Sequential(
            # 76x76 -> 38x38
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 38x38 -> 19x19
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 19x19 -> 10x10
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 10x10 -> 5x5
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )
        
        
        # Project to embedding dimension
        self.projection = nn.Linear(512, embed_dim)
        
    def forward(self, patches):
        # patches: (batch_size, num_patches, 76, 76)
        # coordinates: (batch_size, num_patches, 2)
        
        batch_size, num_patches, h, w = patches.shape
        
        # Reshape for processing
        patches = patches.reshape(batch_size * num_patches, 1, h, w)
        
        # Encode patches
        patch_features = self.patch_encoder(patches)  # (batch_size*num_patches, 512, 1, 1)
        patch_features = patch_features.squeeze(-1).squeeze(-1)  # (batch_size*num_patches, 512)
        patch_features = self.projection(patch_features)  # (batch_size*num_patches, embed_dim)
        
        # Reshape back
        patch_features = patch_features.reshape(batch_size, num_patches, -1)
        
        return patch_features

class ViTStitchingEncoder(nn.Module):
    """Vision Transformer for understanding patch relationships"""
    
    def __init__(self, embed_dim=768):
        super().__init__()
        
        vit_model_full = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_encoder = vit_model_full.encoder 
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Position embeddings for 25 patches + 1 CLS token
        self.position_embeddings = nn.Parameter(torch.randn(1,26, embed_dim))
        
    def forward(self, x):
        # x shape: (batch_size, 25, embed_dim)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embeddings
        x = x + self.position_embeddings.expand(batch_size, -1, -1)
        
        # Pass through ViT
        output = self.vit_encoder(hidden_states=x).last_hidden_state
        output = output[:, 1:, :]  # Exclude CLS token output
        return output  # (batch_size, 25, embed_dim)

class CNNStitchingDecoder(nn.Module):
    """CNN decoder to generate full phase image"""
    
    def __init__(self, embed_dim=768, output_size=256):
        super().__init__()
        self.decoder = nn.Sequential(
            # (batch_size, embed_dim, 5, 5) -> (batch_size, 512, 10, 10)
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # (batch_size, 512, 10, 10) -> (batch_size, 256, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=3), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # (batch_size, 256, 16, 16) -> (batch_size, 128, 32, 32) 
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (batch_size, 128, 32, 32) -> (batch_size, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (batch_size, 64, 64, 64) -> (batch_size, 32, 128, 128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (batch_size, 32, 128, 128) -> (batch_size, 16, 256, 256)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x shape: (batch_size, 25, embed_dim)
        batch_size = x.shape[0]
        x = x.transpose(1, 2)  # (batch_size, embed_dim, 25)
        x = x.reshape(batch_size, -1, 5, 5) # (batch_size, embed_dim, 5, 5)
        
        x = self.decoder(x)
        return x.squeeze(1)
        
class PhaseStitchingNet(nn.Module):
    """Complete phase stitching network"""
    
    def __init__(self, embed_dim=768):
        super().__init__()
        
        self.patch_encoder = PatchEncoder(embed_dim=embed_dim)
        self.vit_encoder = ViTStitchingEncoder(embed_dim=embed_dim)
        self.decoder = CNNStitchingDecoder(embed_dim=embed_dim)
        
    def forward(self, phase_patches):
        # Encode patches with coordinates
        encoded_patches = self.patch_encoder(phase_patches)
        
        # Process with ViT
        processed_features = self.vit_encoder(encoded_patches)
        
        # Decode to full phase
        full_phase = self.decoder(processed_features)
        
        return full_phase

def train_phase_stitching():
    """Training function for PhaseStitchingNet"""
    
    parser = argparse.ArgumentParser(description="Train PhaseStitchingNet")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing simulation data files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="phase-stitching-net", help="Wandb project name")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum change to qualify as an improvement for early stopping in relative terms")
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, config=args)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare data files
    data_files = [
        os.path.join(args.data_dir, f"simulation_data{i}.h5") 
        for i in range(1, 6)
    ]
    
    # Split data for train/validation
    train_files = data_files[:4]  # First 4 files for training
    val_files = data_files[4:]    # Last file for validation
    
    # Create datasets
    train_dataset = PhaseStitchingDataset(train_files)
    val_dataset = PhaseStitchingDataset(val_files)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = PhaseStitchingNet().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (phase_patches, gt_full_phase) in enumerate(train_loader):
            phase_patches = phase_patches.to(device)
            # coordinates = coordinates.to(device)
            gt_full_phase = gt_full_phase.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_full_phase = model(phase_patches)
            
            # Calculate loss
            loss = criterion(pred_full_phase, gt_full_phase)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch_step": epoch + batch_idx / len(train_loader)
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for phase_patches,  gt_full_phase in val_loader:
                phase_patches = phase_patches.to(device)
                gt_full_phase = gt_full_phase.to(device)
                
                pred_full_phase = model(phase_patches)
                loss = criterion(pred_full_phase, gt_full_phase)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": train_loss,
            "avg_val_loss": val_loss,
            "learning_rate": current_lr
        }, step=epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss * (1 - args.min_delta):
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "phase_stitching_best.pth"))
            print(f"Saved best model with validation loss: {val_loss:.6f}")
        else:
            epochs_no_improve +=1
        
        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} after {args.early_stopping_patience} epochs with no improvement.")
            print(f"Best validation loss: {best_val_loss:.6f}")
            wandb.log({"early_stopping_triggered_epoch": epoch})
            break

    wandb.finish()

if __name__ == "__main__":
    train_phase_stitching()