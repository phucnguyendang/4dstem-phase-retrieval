import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import argparse
import os
from typing import Tuple, List

# Import models from other files
from patchRecovery import PatchRecoveryNet, PatchRecoveryDataset
from PhaseStitching import PhaseStitchingNet
import wandb

class End2EndDataset(Dataset):
    """Dataset for end-to-end training"""
    
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
            # Load diffraction patterns (50x50 grid)
            dp_data = f[f"dp_set_{sample_id}"][:]  # Shape: (50, 50, H, W)
            phase_data = f[f"phase_{sample_id}"][:]  # Shape: (256, 256)
        
        # Extract 14x14 DP patches and their corresponding positions
        dp_patches = []
        # For 50x50 grid with 14x14 patches and stride=9, we get 5x5 = 25 patches
        for patch_row in range(5):
            for patch_col in range(5):
                start_row = patch_row * self.stride
                start_col = patch_col * self.stride
                
                # Extract 14x14 DP patch
                dp_patch = dp_data[start_row:start_row+14, start_col:start_col+14]
                dp_patches.append(dp_patch)
        
        # Convert to tensors
        dp_patches = torch.FloatTensor(np.array(dp_patches))  # (25, 14, 14, H, W)
        gt_full_phase = torch.FloatTensor(phase_data)          # (256, 256)
        
        return dp_patches, gt_full_phase

class End2EndModel(nn.Module):
    """End-to-end model combining PatchRecoveryNet and PhaseStitchingNet"""
    
    def __init__(self): 
        super().__init__()
        
        self.patch_recovery = PatchRecoveryNet()
        self.phase_stitching = PhaseStitchingNet()
        
    
    def forward(self, dp_patches):
        """
        Args:
            dp_patches: (batch_size, 25, 14, 14, H, W) - 25 DP patches per sample
        
        Returns:
            full_phase: (batch_size, 256, 256) - reconstructed full phase image
        """
        batch_size, num_patches = dp_patches.shape[:2]
        
        # Process each DP patch through PatchRecoveryNet
        recovered_patches = []
        
        for i in range(num_patches):
            # Get single patch and its coordinates
            single_dp_patch = dp_patches[:, i]  # (batch_size, 14, 14, H, W)
            
            # Recover phase patch
            recovered_patch = self.patch_recovery(single_dp_patch)  # (batch_size, 76, 76)
            recovered_patches.append(recovered_patch)
        
        # Stack recovered patches
        recovered_patches = torch.stack(recovered_patches, dim=1)  # (batch_size, 25, 76, 76)
        # Stitch patches into full phase image
        full_phase = self.phase_stitching(recovered_patches)
        
        return full_phase

def train_end2end():
    """Training function for end-to-end model"""
    
    parser = argparse.ArgumentParser(description="Train End2End Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing simulation data files")
    parser.add_argument("--patch_recovery_checkpoint", type=str, required=True, help="Path to pretrained PatchRecoveryNet")
    parser.add_argument("--phase_stitching_checkpoint", type=str, required=True, help="Path to pretrained PhaseStitchingNet")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_end2end", help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="end2end-model", help="Wandb project name") # Added wandb_project argument
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping (epochs)") # Added early_stopping_patience argument
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change to qualify as an improvement for early stopping in relative terms") # Added min_delta argument
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
    train_dataset = End2EndDataset(train_files)
    val_dataset = End2EndDataset(val_files)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = End2EndModel().to(device) # Initialize model without checkpoints

    # Load pretrained weights for sub-models if provided
    if args.patch_recovery_checkpoint:
        loaded_pr_checkpoint = torch.load(args.patch_recovery_checkpoint, map_location=torch.device('cpu'))
        if isinstance(loaded_pr_checkpoint, dict) and 'model_state_dict' in loaded_pr_checkpoint:
            model.patch_recovery.load_state_dict(loaded_pr_checkpoint['model_state_dict'])
            print(f"Loaded PatchRecoveryNet state_dict from dictionary checkpoint {args.patch_recovery_checkpoint}")
        else:
            model.patch_recovery.load_state_dict(loaded_pr_checkpoint)
            print(f"Loaded PatchRecoveryNet state_dict directly from {args.patch_recovery_checkpoint}")

    if args.phase_stitching_checkpoint:
        loaded_ps_checkpoint = torch.load(args.phase_stitching_checkpoint, map_location=torch.device('cpu'))
        if isinstance(loaded_ps_checkpoint, dict) and 'model_state_dict' in loaded_ps_checkpoint:
            model.phase_stitching.load_state_dict(loaded_ps_checkpoint['model_state_dict'])
            print(f"Loaded PhaseStitchingNet state_dict from dictionary checkpoint {args.phase_stitching_checkpoint}")
        else:
            model.phase_stitching.load_state_dict(loaded_ps_checkpoint)
            print(f"Loaded PhaseStitchingNet state_dict directly from {args.phase_stitching_checkpoint}")
    
    # Loss function and optimizer
    mse_loss = nn.MSELoss() # Use nn.MSELoss directly
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, threshold=0.01) # Updated scheduler
    
    # Training loop
    best_val_loss = float('inf')
    # Add epochs_no_improve for potential early stopping if desired later
    epochs_no_improve = 0 

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (dp_patches, gt_full_phase) in enumerate(train_loader):
            dp_patches = dp_patches.to(device)
            gt_full_phase = gt_full_phase.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_full_phase = model(dp_patches)
            
            # Calculate loss
            total_loss = mse_loss(pred_full_phase, gt_full_phase)  # Calculate loss directly
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "batch_train_loss": total_loss.item(),
                    "learning_rate": current_lr,
                    "epoch_step": epoch + batch_idx / len(train_loader)
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for dp_patches, gt_full_phase in val_loader:
                dp_patches = dp_patches.to(device)
                gt_full_phase = gt_full_phase.to(device)
                
                pred_full_phase = model(dp_patches)
                total_loss = mse_loss(pred_full_phase, gt_full_phase)  # Calculate loss directly
                
                val_loss += total_loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}:")
        print(f"  Train - Total: {train_loss:.6f}") # Adjusted print
        print(f"  Val   - Total: {val_loss:.6f}") # Adjusted print
        
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": train_loss,
            "avg_val_loss": val_loss,
            "learning_rate": current_lr
        }, step=epoch)

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss - args.min_delta * best_val_loss: # Check for improvement based on min_delta
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "end2end_best.pth"))
            print(f"Saved best model with validation loss: {val_loss:.6f}")
            epochs_no_improve = 0 # Reset counter
        else:
            epochs_no_improve += 1 # Increment counter
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
            break # Exit training loop
        
    wandb.finish() # Finish wandb run


if __name__ == "__main__":
    train_end2end()