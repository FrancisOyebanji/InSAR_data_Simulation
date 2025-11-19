# temporal_conv_autoencoder_full.py
# TensorBoard removed and replaced with CSV logging for HPC compatibility.

# Import standard libraries
import os  # For file and directory operations
import math  # For mathematical operations
from typing import Optional, Callable, Tuple, List  # For type hints
import numpy as np  # For numerical computations
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for neural networks
from torch.utils.data import Dataset, DataLoader  # Dataset and batch loading utilities
import matplotlib.pyplot as plt  # For visualization and plotting

# Import custom InSAR dataset
from insar_dataset import create_dataloader  # DataLoader factory for InSAR data

# Set output directory for results
OUT_DIR = "/mnt/data/temporal_ae_outputs"
os.makedirs(OUT_DIR, exist_ok=True)  # Create directory if it doesn't exist

# ------------------------------
# Simple CSV Logger (HPC Safe)
# ------------------------------
class CSVLogger:
    def __init__(self, path):
        self.path = path
        with open(self.path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    def log(self, epoch, train_loss, val_loss):
        with open(self.path, "a") as f:
            f.write(f"{epoch},{train_loss},{val_loss}\n")

    def close(self):
        pass


# ------------------------------
# Model
# ------------------------------
class TemporalConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dem_channels: int = 1,
        num_filters: int = 64,
        encoder_layers: int = 7,  # Changed from 6 to 7 layers
        decoder_layers: int = 4,
        time_kernel: int = 2,
        negative_slope: float = 0.2,
        use_adaptive_pool: bool = True,
    ):
        super().__init__()
        assert encoder_layers == 7 and decoder_layers == 4  # Updated assertion
        self.in_channels = in_channels
        self.dem_channels = dem_channels
        self.num_filters = num_filters
        self.time_kernel = time_kernel
        self.use_adaptive_pool = use_adaptive_pool

        # Build 7 encoder layers with specific pooling schedule:
        # Layers 1-4: (B, 64, 9, H, W) - no pooling
        # Layer 5: (B, 64, 3, H, W) - pool 9→3
        # Layer 6: (B, 64, 3, H, W) - no pooling
        # Layer 7: (B, 64, 1, H, W) - pool 3→1
        self.encoder_convs = nn.ModuleList()
        self.pool_after = [False, False, False, False, True, False, True]  # Pool after layers 5 and 7
        self.pool_kernels = [(1, 1, 1)] * encoder_layers
        self.pool_kernels[4] = (3, 1, 1)  # Layer 5: pool 9->3
        self.pool_kernels[6] = (3, 1, 1)  # Layer 7: pool 3->1
        
        current_in = in_channels
        for i in range(encoder_layers):
            conv = nn.Conv3d(
                in_channels=current_in,
                out_channels=num_filters,
                kernel_size=(time_kernel, 3, 3),
                padding=(time_kernel // 2, 1, 1),
                bias=True,
            )
            self.encoder_convs.append(conv)
            current_in = num_filters
        self.enc_act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # merge conv
        self.merge_conv = nn.Conv2d(
            in_channels=num_filters + dem_channels, out_channels=num_filters, kernel_size=3, padding=1, bias=True
        )
        self.merge_act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # decoder convs (3 layers + final)
        self.decoder_convs = nn.ModuleList()
        dec_in_ch = num_filters
        for i in range(decoder_layers - 1):
            self.decoder_convs.append(nn.Conv2d(dec_in_ch, num_filters, kernel_size=3, padding=1, bias=True))
            dec_in_ch = num_filters
        self.decoder_final = nn.Conv2d(dec_in_ch, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, dem: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("x must be 5D tensor (B,C,T,H,W)")
        b, c, t, h, w = x.shape
        feats = x
        
        # Apply encoder layers with selective temporal pooling
        for i, conv3 in enumerate(self.encoder_convs):
            feats = conv3(feats)
            feats = self.enc_act(feats)
            # Apply pooling after layers 5 and 7 only (when pool_after[i] is True)
            if self.pool_after[i] and feats.shape[2] > 1:
                feats = F.max_pool3d(feats, kernel_size=self.pool_kernels[i])

        # Ensure temporal dimension is reduced to 1
        if feats.shape[2] != 1:
            if self.use_adaptive_pool:
                feats = F.adaptive_avg_pool3d(feats, output_size=(1, h, w))
            else:
                feats = feats.mean(dim=2, keepdim=True)
        feats2d = feats.squeeze(2)

        if dem is not None:
            merged = torch.cat([feats2d, dem], dim=1)
        else:
            zeros = feats2d.new_zeros((b, self.dem_channels, h, w))
            merged = torch.cat([feats2d, zeros], dim=1)

        x2 = self.merge_conv(merged)
        x2 = self.merge_act(x2)
        dec = x2
        for conv2 in self.decoder_convs:
            dec = conv2(dec)
            dec = self.enc_act(dec)
        out = self.decoder_final(dec)
        return out


# ------------------------------
# Real InSAR Data Processing


# Visualization utility for displaying model predictions vs ground truth
def visualize_sample(noisy_seq, dem, target, pred, out_prefix):
    noisy = noisy_seq.detach().cpu().numpy()
    demn = dem.detach().cpu().numpy()
    targ = target.detach().cpu().numpy()
    pr = pred.detach().cpu().numpy()
    C,T,H,W = noisy.shape

    fig, axs = plt.subplots(1, min(6,T)+2, figsize=(12,3))
    for i in range(min(6,T)):
        axs[i].imshow(noisy[0,i], cmap="viridis")
        axs[i].axis("off")
    axs[min(6,T)].imshow(demn[0], cmap="viridis")
    axs[min(6,T)].axis("off")
    axs[min(6,T)+1].imshow(pr[0], cmap="viridis")
    axs[min(6,T)+1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_sample.png", dpi=150)
    plt.close(fig)


# Helper functions for model utilities
def count_parameters(model):
    # Count total trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, path):
    # Save model state and optimizer state for resuming training
    state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}
    torch.save(state, path)


def load_insar_data(data_dir="insar_localized_refined", batch_size=1):
    """
    Load real InSAR data from the insar_localized_refined folder using DataLoader.
    
    Args:
        data_dir (str): Path to the directory containing generated InSAR data
        batch_size (int): Batch size for DataLoader
    
    Returns:
        DataLoader: PyTorch DataLoader with InSAR data
    """
    frames_path = os.path.join(data_dir, "insar_with_refined_localized_deformation.npy")
    dem_path = os.path.join(data_dir, "ground_truth_dem.npy")
    target_path = os.path.join(data_dir, "ground_truth_amplitude.npy")
    mask_path = os.path.join(data_dir, "ground_truth_mask.npy")
    
    # Create and return DataLoader
    return create_dataloader(
        frames_path=frames_path,
        dem_path=dem_path,
        target_path=target_path,
        mask_path=mask_path,
        batch_size=batch_size,
        shuffle=False
    )


# Training / Validation with CSV Logging
# ------------------------------
def train_and_validate_demo(
    device,
    n_epochs=1,
    batch_size=1,
    use_adaptive_pool=True,
    data_dir="insar_localized_refined",
):
    model = TemporalConvAutoencoder(in_channels=1, dem_channels=1, num_filters=64,
                                    encoder_layers=7, decoder_layers=4,
                                    use_adaptive_pool=use_adaptive_pool)
    model = model.to(device)
    print("Model created. Trainable params:", count_parameters(model))

    # Load real InSAR data
    print(f"Loading real InSAR data from '{data_dir}'...")
    train_loader = load_insar_data(data_dir=data_dir, batch_size=batch_size)
    # For real data, use the same loader for validation
    val_loader = load_insar_data(data_dir=data_dir, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # ---- CSV Logger instead of TensorBoard ----
    writer = CSVLogger(os.path.join(OUT_DIR, "training_log.csv"))

    best_val = float("inf")
    for epoch in range(1, n_epochs+1):
        # ---- Train ----
        model.train()
        running = 0.0
        for batch in train_loader:
            # Handle both synthetic (3 outputs) and real data (4 outputs with mask)
            if len(batch) == 4:
                noisy_seq, dem, target, mask = batch
            else:
                noisy_seq, dem, target = batch
            
            noisy_seq, dem, target = noisy_seq.to(device), dem.to(device), target.to(device)
            pred = model(noisy_seq, dem)
            loss = mse(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()*noisy_seq.size(0)

        train_loss = running / len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Handle both synthetic (3 outputs) and real data (4 outputs with mask)
                if len(batch) == 4:
                    noisy_seq, dem, target, mask = batch
                else:
                    noisy_seq, dem, target = batch
                
                noisy_seq, dem, target = noisy_seq.to(device), dem.to(device), target.to(device)
                pred = model(noisy_seq, dem)
                loss = mse(pred, target)
                running_val += loss.item()*noisy_seq.size(0)

        val_loss = running_val / len(val_loader.dataset)

        print(f"Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # log to CSV
        writer.log(epoch, train_loss, val_loss)

        # checkpoint
        ckpt_path = os.path.join(OUT_DIR, f"checkpoint_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(OUT_DIR, "best_checkpoint.pt"))

    writer.close()
    print("Training complete. Outputs written to", OUT_DIR)
    return model


# Entry point - Run the model training demo
if __name__ == "__main__":
    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train with real InSAR data from insar_localized_refined folder
    model = train_and_validate_demo(
        device=device,  # Use selected device
        n_epochs=10,  # Train for 10 epochs
        batch_size=1,  # Batch size 1 for single time series
        use_adaptive_pool=True,  # Use adaptive pooling for better compression
        data_dir="insar_localized_refined"  # Path to generated InSAR data
    )
