import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import transformers
from transformers import ViTModel, ViTConfig
import h5py
import numpy as np
import argparse
import os
from typing import Tuple, List
import math
import wandb
class PatchRecoveryDataset(Dataset):
    """Dataset for patch recovery training"""
    
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
        # Each sample generates multiple 14x14 patches with stride=9
        # For 50x50 grid, we can extract (50-14)//9 + 1 = 5 patches per row/col
        return len(self.samples) * 5 * 5  # 25 patches per sample
    
    def __getitem__(self, idx):
        sample_idx = idx // 25
        patch_idx = idx % 25
        
        file_path, sample_id = self.samples[sample_idx]
        with h5py.File(file_path, 'r') as f:
            # Load diffraction patterns (50x50 grid)
            dp_data = f[f"dp_set_{sample_id}"][:]  # Shape: (50, 50, H, W)
            phase_data = f[f"phase_{sample_id}"][:]  # Shape: (256, 256)
        
        # Calculate patch position
        patch_row = patch_idx // 5
        patch_col = patch_idx % 5
        start_row = patch_row * self.stride
        start_col = patch_col * self.stride
        
        # Extract 14x14 DP patch
        dp_patch = dp_data[start_row:start_row+14, start_col:start_col+14]  # (14, 14, H, W)
        
        # Calculate bright center for ground truth extraction
        bright_center_row = round((start_row*2 + 13) * 2.56) # start_row*2 is the position in Angstroms of start_row
        bright_center_col = round((start_col*2 + 13) * 2.56)
        # Calculate shift for ground truth phase
        probe_gpts_tuple = (256, 256)  # Assuming phase is 256x256
        shift_y = probe_gpts_tuple[0] // 2 - bright_center_row
        shift_x = probe_gpts_tuple[1] // 2 - bright_center_col
        shift = (shift_y, shift_x)
        
        # Roll phase and extract 76x76 central region
        rolled_phase = np.roll(phase_data, shift, axis=(0, 1))
        center_y, center_x = rolled_phase.shape[0] // 2, rolled_phase.shape[1] // 2
        gt_phase_patch = rolled_phase[center_y-38:center_y+38, center_x-38:center_x+38]
        
        # Convert to tensors
        dp_patch = torch.FloatTensor(dp_patch)  # (14, 14, H, W)
        gt_phase_patch = torch.FloatTensor(gt_phase_patch)  # (76, 76)
        
        return dp_patch, gt_phase_patch

class ResNetFeatureExtractor(nn.Module):
    """ResNet34 backbone for DP feature extraction"""
    
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=True)
        
        # Remove the final classification layers (AdaptiveAvgPool2d and Fully Connected layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Modify first conv layer for single channel input because DPs are in grayscale
        original_conv = self.features[0]
        self.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize new conv layer weights = mean of weights across color channels
        with torch.no_grad():
            self.features[0].weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
    
    def forward(self, x):
        # x shape: (batch_size, 14, 14, H, W)
        batch_size, grid_h, grid_w, img_h, img_w = x.shape
        
        # Reshape to process all DPs together
        x = x.reshape(batch_size * grid_h * grid_w, 1, img_h, img_w)
        
        # Extract features
        features = self.features(x)  # (batch_size*196, 512, h', w')
        
        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # (batch_size*196, 512)
        
        # Reshape back to grid
        features = features.reshape(batch_size, grid_h * grid_w, 512)  # (batch_size, 196, 512)
        
        return features


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for spatial understanding, using only the encoder part."""
    
    def __init__(self, input_dim=512, hidden_dim=768):
        super().__init__()        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        vit_model_full = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_encoder = vit_model_full.encoder 
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.position_embeddings = vit_model_full.embeddings.position_embeddings

    def forward(self, x):
        # x shape: (batch_size, num_patches, input_dim), ví dụ (batch_size, 196, 512)
        batch_size = x.shape[0]
        
        # Project to ViT dimension
        projected_x = self.input_projection(x)  # (batch_size, num_patches, hidden_dim)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, hidden_dim)
        embeddings_for_vit = torch.cat((cls_tokens, projected_x), dim=1) # (batch_size, num_patches + 1, hidden_dim)    
        # self.position_embeddings đã có kích thước phù hợp
        final_embeddings = embeddings_for_vit + self.position_embeddings.expand(batch_size, -1, -1)        
        encoder_outputs = self.vit_encoder(hidden_states=final_embeddings)
        
        # Shape: (batch_size, num_patches + 1 , hidden_dim)
        output = encoder_outputs.last_hidden_state 
        output = output[:, 1:, :]  # Bỏ CLS token, chỉ lấy các patch
        return output
    
class CNNDecoder(nn.Module):
    """CNN decoder to generate phase patch"""
    
    def __init__(self, input_dim=768, output_size=76): # output_size is effectively 76
        super().__init__()
        
        # (B, 196, 768) -> (B, 196, 512)
        self.feature_projection = nn.Linear(input_dim, 512) 
        
        # The 196 patches can be seen as a 14x14 grid.
        # We will reshape to (B, 512, 14, 14) in the forward pass.
        # Initial spatial size for decoder input will be 14x14.
        
        # Decoder layers to upsample from 14x14 to 76x76
        self.decoder_layers = nn.Sequential(
            # Input: (B, 512, 14, 14)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Output: (B, 256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Input: (B, 256, 28, 28)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (B, 128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Input: (B, 128, 56, 56)
            # To upsample from 56x56 to 76x76 with stride 1:
            # output_H = (input_H - 1)*stride + kernel_size - 2*padding
            # 76 = (56 - 1)*1 + K - 2*P  => K - 2P = 21. Choose K=21, P=0.
            nn.ConvTranspose2d(128, 1, kernel_size=21, stride=1, padding=0) # Output: (B, 1, 76, 76)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 196, 768)
        batch_size = x.shape[0]
        num_patches = x.shape[1] 
        
        # Project features: (batch_size, 196, 768) -> (batch_size, 196, 512)
        x = self.feature_projection(x)
        
        # reshape to (B, 512, 14, 14)
        patch_grid_size = int(math.sqrt(num_patches)) # Should be 14
        # (batch_size, 196, 512) -> (batch_size, 512, 196) -> (batch_size, 512, 14, 14)
        x = x.transpose(1, 2).reshape(batch_size, 512, patch_grid_size, patch_grid_size)
        
        # Pass through decoder
        x = self.decoder_layers(x)  # (batch_size, 1, 76, 76)
        
        return x.squeeze(1)  # (batch_size, 76, 76)

class PatchRecoveryNet(nn.Module):
    """Complete patch recovery network"""
    
    def __init__(self):
        super().__init__()
        
        self.resnet_extractor = ResNetFeatureExtractor()
        self.vit_encoder = ViTEncoder()
        self.cnn_decoder = CNNDecoder()
        
    def forward(self, dp_patch):
        # Extract features from DPs
        features = self.resnet_extractor(dp_patch)
        
        # Encode with ViT
        encoded_features = self.vit_encoder(features)
        
        # Decode to phase patch
        phase_patch = self.cnn_decoder(encoded_features)
        
        return phase_patch

def train_patch_recovery():
    """Training function for PatchRecoveryNet"""
    
    parser = argparse.ArgumentParser(description="Train PatchRecoveryNet")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing simulation data files. Defaults to current directory if not provided.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="patch-recovery-net", help="Wandb project name")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum change to qualify as an improvement for early stopping in relative terms")

    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, config=args)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for training. No CUDA device found.")
    wandb.config.update({"device": str(device)})
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare data files
    data_files = [
        os.path.join(args.data_dir, f"simulation_data{i}.h5") 
        for i in range(1, 6) # Assuming 5 simulation files as in original code
    ]
    
    # Split data for train/validation
    # Ensure there's at least one validation file if data_files is short
    if len(data_files) < 2:
        raise ValueError("Not enough data files for train/validation split. Need at least 2.")
    
    num_val_files = 1
    train_files = data_files[:-num_val_files]
    val_files = data_files[-num_val_files:]

    if not train_files:
        raise ValueError("No files allocated for training. Check data_dir and file naming.")
    if not val_files:
        raise ValueError("No files allocated for validation. Check data_dir and file naming.")

    print(f"Training files: {train_files}")
    print(f"Validation files: {val_files}")
    
    # Create datasets
    train_dataset = PatchRecoveryDataset(train_files)
    val_dataset = PatchRecoveryDataset(val_files)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = PatchRecoveryNet().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True,threshold=0.01) # Reduced patience for scheduler
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss_epoch = 0.0
        
        for batch_idx, (dp_patch, gt_phase) in enumerate(train_loader):
            dp_patch = dp_patch.to(device)
            gt_phase = gt_phase.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_phase = model(dp_patch)
            
            # Calculate loss
            loss = criterion(pred_phase, gt_phase)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{args.epochs-1}, Batch {batch_idx}/{len(train_loader)-1}, Train Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch_step": epoch + batch_idx / len(train_loader) # For finer-grained step logging
                })
        
        avg_train_loss = train_loss_epoch / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_epoch = 0.0
        
        with torch.no_grad():
            for dp_patch, gt_phase in val_loader:
                dp_patch = dp_patch.to(device)
                gt_phase = gt_phase.to(device)
                
                pred_phase = model(dp_patch)
                loss = criterion(pred_phase, gt_phase)
                val_loss_epoch += loss.item()
        
        avg_val_loss = val_loss_epoch / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{args.epochs-1}: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
        
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "learning_rate": current_lr
        }, step=epoch)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if best_val_loss == float('inf') or avg_val_loss < best_val_loss * (1-args.min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(args.save_dir, "patch_recovery_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {avg_val_loss:.6f} at epoch {epoch}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} after {args.early_stopping_patience} epochs with no improvement.")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
            
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    train_patch_recovery()