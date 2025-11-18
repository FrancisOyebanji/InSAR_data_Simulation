# temporal_conv_autoencoder_full.py
# TensorBoard removed and replaced with CSV logging for HPC compatibility.

import os
import math
from typing import Optional, Callable, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/data/temporal_ae_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

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
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        time_kernel: int = 2,
        negative_slope: float = 0.2,
        use_adaptive_pool: bool = True,
    ):
        super().__init__()
        assert encoder_layers == 6 and decoder_layers == 4
        self.in_channels = in_channels
        self.dem_channels = dem_channels
        self.num_filters = num_filters
        self.time_kernel = time_kernel
        self.use_adaptive_pool = use_adaptive_pool

        self.encoder_convs = nn.ModuleList()
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
        for conv3 in self.encoder_convs:
            feats = conv3(feats)
            feats = self.enc_act(feats)
            if self.use_adaptive_pool and feats.shape[2] > 1:
                feats = F.max_pool3d(feats, kernel_size=(2, 1, 1))

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
# Synthetic Spatial Patterns
# ------------------------------
def synth_unipolar(shape: Tuple[int,int], amplitude=1.0, center=None, sigma=5.0):
    H, W = shape
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    cy = H/2 if center is None else center[0]
    cx = W/2 if center is None else center[1]
    gauss = amplitude * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma*sigma))
    return gauss.astype(np.float32)

def synth_dipole(shape, amplitude=1.0, separation=10, sigma=3.0):
    H, W = shape
    p1 = synth_unipolar(shape, amplitude=amplitude, center=(H//2, W//2 - separation//2), sigma=sigma)
    p2 = synth_unipolar(shape, amplitude=-amplitude, center=(H//2, W//2 + separation//2), sigma=sigma)
    return (p1 + p2).astype(np.float32)


# ------------------------------
# Flexible Dataset
# ------------------------------
class SpatioTemporalDatasetFlexible(Dataset):
    def __init__(
        self,
        sequences=None,
        dems=None,
        synth=True,
        n_samples=100,
        frame_shape=(1,8,64,64),
        dem_channels=1,
        noise_std=0.02,
        deformation_type="sum",
        dipole_params=None
    ):
        self.synth = synth
        self.sequences = sequences
        self.dems = dems
        self.n = n_samples if synth else (len(sequences) if sequences is not None else 0)
        self.C, self.T, self.H, self.W = frame_shape
        self.dem_ch = dem_channels
        self.noise_std = noise_std
        self.deformation_type = deformation_type
        self.dipole_params = dipole_params or {"amplitude": 0.5, "separation": 8, "sigma": 3.0}

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.synth:
            seq = np.random.randn(self.C, self.T, self.H, self.W).astype(np.float32) * 0.05
            t = np.arange(self.T).astype(np.float32)
            temporal_profile = np.sin(2*np.pi*(t / max(1,self.T))) * 0.1
            for ti in range(self.T):
                seq[0,ti] += temporal_profile[ti]

            if self.deformation_type == "sum":
                base = synth_unipolar((self.H,self.W), amplitude=0.1, sigma=8.0)
                for ti in range(self.T):
                    seq[0,ti] += base * (ti/self.T)
                target = seq.sum(axis=1, keepdims=True)

            elif self.deformation_type == "unipolar":
                pat = synth_unipolar((self.H,self.W), amplitude=self.dipole_params["amplitude"],
                                     sigma=self.dipole_params["sigma"])
                for ti in range(self.T):
                    seq[0,ti] += pat * ((ti+1)/self.T)
                target = pat[np.newaxis,:,:]

            elif self.deformation_type == "dipole":
                pat = synth_dipole((self.H,self.W),
                                   amplitude=self.dipole_params["amplitude"],
                                   separation=self.dipole_params["separation"],
                                   sigma=self.dipole_params["sigma"])
                for ti in range(self.T):
                    seq[0,ti] += pat * ((ti+1)/self.T)
                target = pat[np.newaxis,:,:]

            dem = np.random.randn(self.dem_ch, self.H, self.W).astype(np.float32) * 0.05

        else:
            seq = self.sequences[idx].astype(np.float32)
            dem = self.dems[idx].astype(np.float32)
            target = seq.sum(axis=1, keepdims=True)

        noisy = seq + np.random.randn(*seq.shape).astype(np.float32) * self.noise_std
        return torch.from_numpy(noisy), torch.from_numpy(dem), torch.from_numpy(target)


# ------------------------------
# Visualization
# ------------------------------
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


# ------------------------------
# Helper functions
# ------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, path):
    state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}
    torch.save(state, path)


# ------------------------------
# Training / Validation with CSV Logging
# ------------------------------
def train_and_validate_demo(
    device,
    n_epochs=1,
    batch_size=8,
    synth_deformation="sum",
    use_adaptive_pool=True,
):
    model = TemporalConvAutoencoder(in_channels=1, dem_channels=1, num_filters=64,
                                    use_adaptive_pool=use_adaptive_pool)
    model = model.to(device)
    print("Model created. Trainable params:", count_parameters(model))

    train_ds = SpatioTemporalDatasetFlexible(synth=True, n_samples=80,
                                             frame_shape=(1,8,64,64),
                                             dem_channels=1, noise_std=0.02,
                                             deformation_type=synth_deformation)
    val_ds = SpatioTemporalDatasetFlexible(synth=True, n_samples=20,
                                           frame_shape=(1,8,64,64),
                                           dem_channels=1, noise_std=0.02,
                                           deformation_type=synth_deformation)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # ---- CSV Logger instead of TensorBoard ----
    writer = CSVLogger(os.path.join(OUT_DIR, "training_log.csv"))

    best_val = float("inf")
    for epoch in range(1, n_epochs+1):
        # ---- Train ----
        model.train()
        running = 0.0
        for noisy_seq, dem, target in train_loader:
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
            for noisy_seq, dem, target in val_loader:
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

        # visualization
        sample_noisy, sample_dem, sample_targ = val_ds[0]
        pred_sample = model(sample_noisy.unsqueeze(0).to(device),
                            sample_dem.unsqueeze(0).to(device))
        visualize_sample(sample_noisy, sample_dem, sample_targ,
                         pred_sample.cpu()[0],
                         os.path.join(OUT_DIR, f"epoch{epoch}"))

    writer.close()
    print("Training complete. Outputs written to", OUT_DIR)
    return model


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_and_validate_demo(
        device=device,
        n_epochs=1,
        batch_size=8,
        synth_deformation="dipole",
        use_adaptive_pool=True
    )
