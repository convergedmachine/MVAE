
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreyFace VAE vs MVAE visualization (z=32)
- Loads weights:
    * freyface_zdim32_VAE.pkl
    * freyface_zdim32_MVAE.pkl
- Produces:
    1) 2D latent factor sweep on (z_i, z_j) with other dims=0  (two scenarios: CDF grid vs. linear grid)
    2) Random generations (same noise) for VAE & MVAE
    3) Reconstructions on a small batch (original vs VAE vs MVAE) + per-model MSE
All figures saved under results/FreyFace (configurable).
"""
import os
import sys
import math
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from scipy.stats import norm
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat


class FreyFaceDataset(torch.utils.data.Dataset):
    # data_file: available at https://cs.nyu.edu/~roweis/data/frey_rawface.mat
    data_file = 'frey_rawface.mat'

    def __init__(self, root, transform=None):
        super(FreyFaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        if not self._check_exists():
            raise RuntimeError('Dataset do not found in the directory \"{}\". \nYou can download FreyFace '
                               'dataset from https://cs.nyu.edu/~roweis/data/frey_rawface.mat '.format(self.root))
        self.data = loadmat(os.path.join(self.root, self.data_file))['ff'].T

    def __getitem__(self, index):
        img = self.data[index].reshape(28, 20)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.data_file))

# ----------------------------- Robust imports -----------------------------
# Expect the user's project layout to have models.py exporting VAE/MVAE.
# We also try a few common dataset module paths for FreyFaceDataset.
def _try_import_models():
    candidates = [
        ("models", "VAE", "MVAE"),
        ("src.models", "VAE", "MVAE"),
    ]
    last_err = None
    for mod, vae_name, mvae_name in candidates:
        try:
            _m = __import__(mod, fromlist=[vae_name, mvae_name])
            VAE = getattr(_m, vae_name)
            MVAE = getattr(_m, mvae_name, None)
            return VAE, MVAE
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import VAE/MVAE from your project. Last error: {last_err}")

# ----------------------------- Utility helpers -----------------------------
def produce_z_values(nrows: int, ncolumes: int, scenario: int = 1) -> np.ndarray:
    if scenario == 1:
        cdf_range1 = np.linspace(1e-5, 1 - 1e-5, ncolumes)
        cdf_range2 = np.linspace(1 - 1e-5, 1e-5, nrows)
        mat_z1, mat_z2 = np.meshgrid(norm.ppf(cdf_range1), norm.ppf(cdf_range2))
        z_values = np.concatenate((mat_z1.reshape(-1, 1), mat_z2.reshape(-1, 1)), axis=1)
        return z_values
    elif scenario == 2:
        z_range1 = np.linspace(-4.0, 4.0, ncolumes)
        z_range2 = np.linspace(4.0, -4.0, nrows)
        mat_z1, mat_z2 = np.meshgrid(z_range1, z_range2)
        z_values = np.concatenate((mat_z1.reshape(-1, 1), mat_z2.reshape(-1, 1)), axis=1)
        return z_values
    else:
        raise ValueError('The argument "scenario" must be an integer from the set {1, 2}.')

def safe_device(dev_arg: str) -> torch.device:
    if dev_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_arg)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

@torch.no_grad()
def safe_decode(model, z: torch.Tensor) -> torch.Tensor:
    out = model.decode(z)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

@torch.no_grad()
def safe_forward_reconst(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    # handle various return conventions
    if isinstance(out, (list, tuple)):
        last = out[-1]
        if isinstance(last, (list, tuple)):
            return last[0]
        return last
    return out

def to_img_grid(x: torch.Tensor, nrow: int) -> torch.Tensor:
    # x is [N, 1, 28, 20] for FreyFace
    return make_grid(x, nrow=nrow)

# ----------------------------- Core visualization -----------------------------
def do_factor_sweep(model, latent_size: int, dim_i: int, dim_j: int,
                    nrows: int, ncols: int, scenario: int, device: torch.device,
                    out_path: str):
    """
    Vary (z[dim_i], z[dim_j]) on a 2D grid; all other dims=0.
    """
    z2 = produce_z_values(nrows=nrows, ncolumes=ncols, scenario=scenario)  # [nrows*ncols, 2]
    z = torch.zeros((z2.shape[0], latent_size), dtype=torch.float32, device=device)
    z[:, dim_i] = torch.from_numpy(z2[:, 0]).to(device)
    z[:, dim_j] = torch.from_numpy(z2[:, 1]).to(device)
    x_hat = safe_decode(model, z).view(-1, 1, 28, 20).cpu()
    save_image(x_hat, out_path, nrow=ncols)

def compare_random_generations(model_vae, model_mvae, latent_size: int, n: int,
                               device: torch.device, out_path_vae: str, out_path_mvae: str):
    torch.manual_seed(0)
    z = torch.randn(n, latent_size, device=device)
    xhat_vae = safe_decode(model_vae, z).view(-1, 1, 28, 20).cpu()
    xhat_mvae = safe_decode(model_mvae, z).view(-1, 1, 28, 20).cpu()
    save_image(xhat_vae, out_path_vae, nrow=int(math.sqrt(n)) if int(math.sqrt(n))**2 == n else 8)
    save_image(xhat_mvae, out_path_mvae, nrow=int(math.sqrt(n)) if int(math.sqrt(n))**2 == n else 8)

@torch.no_grad()
def compare_reconstructions(model_vae, model_mvae, loader, device: torch.device,
                            out_path_triplet: str, max_batches: int = 1) -> Tuple[float, float]:
    """
    Save a single grid showing [original | VAE | MVAE]. Return (mse_vae, mse_mvae).
    """
    model_vae.eval(); model_mvae.eval()
    mse_vae_acc, mse_mvae_acc, n_total = 0.0, 0.0, 0
    triplet_imgs = []

    for b_idx, batch in enumerate(loader):
        # batch may be (x, y) or just x depending on the dataset class
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)  # [B,1,28,20] expected
        B = x.size(0)
        x_flat = x.view(B, -1)  # 560 dims

        rec_vae = safe_forward_reconst(model_vae, x_flat).view(B, 1, 28, 20)
        rec_mvae = safe_forward_reconst(model_mvae, x_flat).view(B, 1, 28, 20)

        # per-batch MSE (pixel-space, average)
        mse_vae = torch.mean((x.view(B, -1) - rec_vae.view(B, -1))**2).item()
        mse_mvae = torch.mean((x.view(B, -1) - rec_mvae.view(B, -1))**2).item()
        mse_vae_acc += mse_vae * B
        mse_mvae_acc += mse_mvae * B
        n_total += B

        # interleave: [orig, vae, mvae] for first K
        K = min(24, B)
        row = torch.cat([x[:K].cpu(), rec_vae[:K].cpu(), rec_mvae[:K].cpu()], dim=0)
        triplet_imgs.append(row)

        if b_idx + 1 >= max_batches:
            break

    # stack rows and save as grid
    triplet = torch.cat(triplet_imgs, dim=0)
    grid = make_grid(triplet, nrow=K)
    torchvision.utils.save_image(grid, out_path_triplet)

    return mse_vae_acc / max(n_total, 1), mse_mvae_acc / max(n_total, 1)

def maybe_encode_scatter(model, loader, device: torch.device, out_path: str, title: str):
    """
    Optional: if the model has .encode(x)->(mu, ...), scatter PCA(mu) to 2D.
    If not available, this silently skips.
    """
    try:
        from sklearn.decomposition import PCA
    except Exception:
        print("sklearn not available: skipping latent scatter.")
        return

    zs = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            x = x.view(x.size(0), -1)
            if not hasattr(model, "encode"):
                print("Model has no .encode(); skipping scatter.")
                return
            enc = model.encode(x)
            if isinstance(enc, (list, tuple)):
                mu = enc[0]
            else:
                mu = enc
            zs.append(mu.detach().cpu().numpy())
            if b_idx > 20:  # enough samples for a feel
                break
    if not zs:
        return
    Z = np.concatenate(zs, axis=0)
    Z2 = PCA(n_components=2).fit_transform(Z)

    plt.figure(figsize=(5, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], s=5, alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-vae", type=str, default="trained_parameters/freyface_zdim32_VAE.pkl")
    parser.add_argument("--weights-mvae", type=str, default="trained_parameters/freyface_zdim32_MVAE.pkl")
    parser.add_argument("--data-root", type=str, default="./data_folder/FreyFace")
    parser.add_argument("--results-dir", type=str, default="results/FreyFace")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=14)
    parser.add_argument("--dim-i", type=int, default=0, help="First latent dimension to sweep")
    parser.add_argument("--dim-j", type=int, default=1, help="Second latent dimension to sweep")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument("--n-generate", type=int, default=64)
    parser.add_argument("--recon-batches", type=int, default=1)
    args = parser.parse_args()

    device = safe_device(args.device)
    ensure_dir(args.results_dir)

    # Load models
    VAE, MVAE = _try_import_models()
    vae = VAE(input_size=560, hidden_size=500, latent_size=args.latent_size).to(device)
    mvae = MVAE(input_size=560, hidden_size=500, latent_size=args.latent_size).to(device) if MVAE is not None else None

    vae.load_state_dict(torch.load(args.weights_vae, map_location=device), strict=True)
    mvae.load_state_dict(torch.load(args.weights_mvae, map_location=device), strict=True)

    vae.eval(); mvae.eval()

    # Data
    tfm = transforms.ToTensor()
    dataset = FreyFaceDataset(root=args.data_root, transform=tfm)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    # 1) Factor sweep on two chosen latent dims
    sweep_vae_path  = os.path.join(args.results_dir, f"sweep_VAE_d{args.latent_size}_i{args.dim_i}_j{args.dim_j}_sc{args.scenario}.png")
    sweep_mvae_path = os.path.join(args.results_dir, f"sweep_MVAE_d{args.latent_size}_i{args.dim_i}_j{args.dim_j}_sc{args.scenario}.png")
    do_factor_sweep(vae,  args.latent_size, args.dim_i, args.dim_j, args.grid_rows, args.grid_cols, args.scenario, device, sweep_vae_path)
    do_factor_sweep(mvae, args.latent_size, args.dim_i, args.dim_j, args.grid_rows, args.grid_cols, args.scenario, device, sweep_mvae_path)

    # 2) Random generations with shared noise
    gen_vae_path  = os.path.join(args.results_dir, f"gen_VAE_d{args.latent_size}.png")
    gen_mvae_path = os.path.join(args.results_dir, f"gen_MVAE_d{args.latent_size}.png")
    compare_random_generations(vae, mvae, args.latent_size, args.n_generate, device, gen_vae_path, gen_mvae_path)

    # 3) Reconstructions (original | VAE | MVAE) + report MSE
    triplet_path = os.path.join(args.results_dir, f"recon_triplet_d{args.latent_size}.png")
    mse_vae, mse_mvae = compare_reconstructions(vae, mvae, loader, device, triplet_path, max_batches=args.recon_batches)
    print(f"[Recon MSE] VAE: {mse_vae:.6f} | MVAE: {mse_mvae:.6f}")
    with open(os.path.join(args.results_dir, "recon_mse.txt"), "w") as f:
        f.write(f"VAE\t{mse_vae:.6f}\nMVAE\t{mse_mvae:.6f}\n")

    # 4) Optional: latent scatter via encoder means
    try:
        scatter_vae = os.path.join(args.results_dir, f"latent_scatter_VAE_d{args.latent_size}.png")
        scatter_mvae = os.path.join(args.results_dir, f"latent_scatter_MVAE_d{args.latent_size}.png")
        maybe_encode_scatter(vae, loader, device, scatter_vae, f"VAE (z={args.latent_size})")
        maybe_encode_scatter(mvae, loader, device, scatter_mvae, f"MVAE (z={args.latent_size})")
    except Exception as e:
        print(f"Latent scatter skipped: {e}")

    print("Done. Outputs saved to:", args.results_dir)


if __name__ == "__main__":
    main()
