import os
import math
import argparse
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from models import VAE, MVAE
from utils import make_gif, plot_elbocurve

import csv
import time
import numpy as np

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, log_loss
from collections import defaultdict
import json
import warnings
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

def save_curves(curves_dir: str, dataset: str, model: str, latent: int, seed: int,
                train_elbo_hist: list[float], test_elbo_hist: list[float],
                train_mse_hist: list[float], test_mse_hist: list[float]):
    """Save training curves (ELBO and optionally MSE) to JSON, CSV, and NPZ files.
    1) curves_dir: directory to save curves
    2) dataset: name of dataset (e.g., "mnist")
    3) model: name of model (e.g., "vae")
    4) latent: latent dimensionality
    5) seed: random seed used
    6) train_elbo_hist: list of training ELBO values per epoch
    7) test_elbo_hist: list of test ELBO values per epoch
    8) train_mse_hist: optional list of training MSE values per epoch
    9) test_mse_hist: optional list of test MSE values per epoch
    """
    os.makedirs(curves_dir, exist_ok=True)
    run_meta = {
        "dataset": dataset,
        "model": model.lower(),
        "latent_dim": int(latent),
        "seed": int(seed),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_epochs": len(train_elbo_hist),
    }

    payload = {
        "meta": run_meta,
        "epochs": list(range(1, len(train_elbo_hist) + 1)),
        "train_elbo": list(map(float, train_elbo_hist)),
        "test_elbo": list(map(float, test_elbo_hist)),
        "train_mse": list(map(float, train_mse_hist)),
        "test_mse": list(map(float, test_mse_hist))
    }

    stem = f"{run_meta['dataset']}_{model.lower()}_d{latent}_s{seed}"
    json_path = os.path.join(curves_dir, f"{stem}.json")

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[saved curves] {json_path}")


@torch.no_grad()
def gather_elbo_mse(model, loader, device, input_size):
    """
    Compute average ELBO and MSE (per pixel) across the dataset.
    Works for both diagonal- and full-covariance VAE variants.
    
    Args:
        model: model with forward(x) -> (mu, logvar_or_L, recon)
        loader: DataLoader providing (x, y)
        device: torch.device('cuda' or 'cpu')
        input_size: flattened input dimension, e.g. 784 for MNIST
        
    Returns:
        dict with:
          - 'elbo': mean ELBO per sample
          - 'mse_per_pixel': mean MSE per pixel
    """
    model.eval()
    total_elbo = 0.0
    total_mse = 0.0
    n_batches = 0
    total_pixels = 0

    for xb, _ in loader:
        x = xb.to(device).view(-1, input_size)

        # Forward pass — adapt automatically
        out = model(x)
        if len(out) == 3:
            mu, var_like, recon = out
        else:
            raise ValueError("Model forward must return (mu, logvar/L, recon)")

        # Compute ELBO (already normalized per batch)
        elbo = model.compute_loss(x, recon, mu, var_like)

        # Compute MSE per pixel
        mse = F.mse_loss(recon, x, reduction="sum")
        total_mse += mse.item()
        total_elbo += elbo.item()
        n_batches += 1
        total_pixels += x.numel()

    avg_elbo = total_elbo / max(1, n_batches)
    avg_mse = total_mse / max(1, total_pixels)

    return {"elbo": avg_elbo, "mse_per_pixel": avg_mse}


@torch.no_grad()
def _extract_latents(model, loader, device, input_size):
    model.eval()
    Z, Y = [], []
    for xb, yb in loader:
        x = xb.to(device).view(-1, input_size)
        mu, _, _ = model(x)
        Z.append(mu.detach().cpu().numpy())  # use μ as representation
        Y.append(np.asarray(yb))
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0) if len(Y) and Y[0] is not None else None
    return Z, Y

def _ece(probs, y_true, n_bins=15):
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (confidences >= lo) & (confidences < hi)
        if sel.any():
            acc = correct[sel].mean()
            conf = confidences[sel].mean()
            ece += (sel.mean()) * abs(acc - conf)
    return float(ece)

def evaluate_model(model, latent, train_loader, test_loader, device, input_size: int, results_dir: str, model_name: str, dataset_name: str, seed: int):
    os.makedirs(results_dir, exist_ok=True)

    # 1) Reconstruction + ELBO
    train_elbo_mse  = gather_elbo_mse(model, train_loader, device, input_size)
    train_elbo = train_elbo_mse["elbo"]
    train_msepp = train_elbo_mse["mse_per_pixel"]

    test_elbo_mse  = gather_elbo_mse(model, test_loader,  device, input_size)
    test_elbo = test_elbo_mse["elbo"]
    test_msepp = test_elbo_mse["mse_per_pixel"]

    # 2) Latents (probe & clustering only if labels are meaningful)
    Ztr, Ytr = _extract_latents(model, train_loader, device, input_size)
    Zte, Yte = _extract_latents(model, test_loader,  device, input_size)

    metrics = {
        "model": model_name,
        "recon/mse_per_pixel": test_msepp,   # ↓
        "elbo/train": train_elbo,            # ↑
        "elbo/test": test_elbo,              # ↑
        "mse/train": train_msepp,          # ↓
        "mse/test": test_msepp,            # ↓
    }

    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="saga", max_iter=500, multi_class="multinomial", n_jobs=-1, verbose=0
    ).fit(Ztr, Ytr)
    yhat = clf.predict(Zte)
    acc = accuracy_score(Yte, yhat)

    proba = clf.predict_proba(Zte)
    nll = log_loss(Yte, proba)  # ↓
    onehot = np.eye(proba.shape[1])[Yte]
    brier = float(np.mean(np.sum((proba - onehot)**2, axis=1)))
    ece = _ece(proba, Yte, n_bins=15)

    # Choose k sensibly: use number of unique labels
    k = int(np.unique(Yte).size)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    clu = km.fit_predict(Zte)
    nmi = normalized_mutual_info_score(Yte, clu, average_method="arithmetic")
    ari = adjusted_rand_score(Yte, clu)

    metrics.update({
        "probe/acc": acc,                    # ↑
        "probe/nll": nll,                    # ↓
        "probe/brier": brier,                # ↓
        "probe/ece": ece,                    # ↓
        "cluster/nmi": nmi,                  # ↑
        "cluster/ari": ari,                  # ↑
    })

    with open(os.path.join(results_dir, f"{dataset_name}_{model_name}_d{latent}_s{seed}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== Evaluation Summary ====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>22s}: {v:.6f}")
        else:
            print(f"{k:>22s}: {v}")
    print("============================\n")
    return metrics

def parse_args():
    """
    Argument parser for training VAE or MVAE models
    across Larochelle, MNIST, Fashion-MNIST, and CIFAR datasets.
    """
    p = argparse.ArgumentParser(
        description="Train VAE or MVAE on MNIST variants, Fashion-MNIST, or CIFAR datasets"
    )

    # ---- Dataset & Model Options ----
    p.add_argument(
        "--dataset", type=str, default="mnist_basic",
        choices=[
            "mnist_basic",
            "mnist_rotated",
            "mnist_background_images",
            "mnist_background_random",
            "fashion_mnist",
            "cifar10",
            "cifar100",
            "freyface"
        ],
        help=("Dataset to use: Larochelle MNIST variants, Fashion-MNIST, Frey Face or CIFAR-10/100.")
    )

    p.add_argument(
        "--model", type=str, default="VAE", choices=["VAE", "MVAE"],
        help="Which model architecture to train (VAE or MVAE)."
    )

    # ---- Training Parameters ----
    p.add_argument("--latent-size", type=int, default=32,
                   help="Dimensionality of latent space z (e.g. {2 4 16 32 256 512}).")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Mini-batch size for training and evaluation.")
    p.add_argument("--epochs", type=int, default=200,
                   help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for AdamW optimizer.")
    p.add_argument("--seed", type=int, default=777,
                   help="Random seed for reproducibility.")

    # ---- Data & Results Paths ----
    p.add_argument(
        "--data-root", type=str, default="./data_folder",
        help=("Root folder for datasets. Subdirectories expected: "
              "MNIST/, FASHION_MNIST/, CIFAR10/, CIFAR100/, etc.")
    )
    p.add_argument(
        "--results-dir", type=str, default="./results",
        help=("Root directory to save training/evaluation results. "
              "Each dataset will have its own subfolder.")
    )
    p.add_argument(
        "--params-dir", type=str, default="./trained_parameters",
        help="Directory to store trained model parameters (.pkl)."
    )

    # ---- Loader & Split Options ----
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of subprocesses for DataLoader.")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="Fraction of data reserved for testing.")

    # ---- Evaluation Mode ----
    p.add_argument(
        "--evaluate", action="store_true",
        help="If set, skips training and runs model evaluation only using pre-trained parameters."
    )    

    return p.parse_args()

# ---------------------------
# Dataset loading (Larochelle + Torchvision)
# ---------------------------
def load_dataset(name: str, root: str = "data_folder"):
    """
    Returns:
        X (np.ndarray, float32, shape [N, D]), y (np.ndarray, int64, shape [N])
    Notes:
      - Torchvision datasets are concatenated (train+test), flattened, and left unnormalized.
        Your preprocessing pipeline (colnorm / PCA) will handle scaling.
      - Larochelle datasets use latent_structure_task() and labels from the dataset object.
    """
    import numpy as np

    # Larochelle et al. 2007
    from datasets.larochelle_etal_2007.dataset import (
        MNIST_Basic, MNIST_BackgroundImages, MNIST_BackgroundRandom, MNIST_Rotated
    )
    larochelle_map = {
        "mnist_basic": MNIST_Basic,
        "mnist_rotated": MNIST_Rotated,
        "mnist_background_images": MNIST_BackgroundImages,
        "mnist_background_random": MNIST_BackgroundRandom,
    }

    if name in larochelle_map:
        D = larochelle_map[name]()
        X = D.latent_structure_task().astype(np.float32)
        y = D._labels.copy().astype(np.int64)
        # Ensure 2D (N, D)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X, y

    # Torchvision datasets
    import torchvision

    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False)
        testset  = torchvision.datasets.CIFAR10(root=root, train=False, download=False)
        X = np.concatenate([trainset.data, testset.data], axis=0)            # (N, 32, 32, 3), uint8
        y = np.array(trainset.targets + testset.targets, dtype=np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0             # (N, 3072)
        return X, y

    if name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=False)
        testset  = torchvision.datasets.CIFAR100(root=root, train=False, download=False)
        X = np.concatenate([trainset.data, testset.data], axis=0)            # (N, 32, 32, 3), uint8
        y = np.array(trainset.targets + testset.targets, dtype=np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0             # (N, 3072)
        return X, y

    if name == "fashion_mnist":
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=False)
        testset  = torchvision.datasets.FashionMNIST(root=root, train=False, download=False)
        # .data is a torch.Tensor (N, 28, 28), uint8
        X = np.concatenate([trainset.data.numpy(), testset.data.numpy()], axis=0)
        y = np.concatenate([trainset.targets.numpy(), testset.targets.numpy()], axis=0).astype(np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0              # (N, 784)
        return X, y
    if name == "freyface":
        entire_dataset = FreyFaceDataset(root=root + '/FreyFace', transform=transforms.ToTensor())
        X = torch.stack([entire_dataset[i] for i in range(len(entire_dataset))], dim=0)
        X = X.view(len(entire_dataset), -1).numpy().astype(np.float32)
        y = np.zeros(len(entire_dataset), dtype=np.int64)
        return X, y        

    raise ValueError(f"Unknown dataset: {name}")

def build_dataloaders(args):
    """
    Uses `load_dataset(name, root)` that returns flat X [N, D] float32 and y [N] int64.
    - Stratified split (per class) with args.test_frac (default 0.2)
    - If args.flatten_mlp is False, tries to reshape to image tensors:
        * CIFAR10/100 -> [N, 3, 32, 32]
        * MNIST-like (Larochelle variants + Fashion-MNIST) -> [N, 1, 28, 28]
      Otherwise keeps flat vectors [N, D].
    - Returns: train_loader, test_loader, input_size, dataset_name
      where input_size is int D if flattened, or tuple [C, H, W] if not.
    """
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # ---- Config & defaults
    torch.manual_seed(getattr(args, "seed", 0))
    rng = np.random.default_rng(getattr(args, "seed", 0))

    dataset_name = args.dataset.lower()
    data_root    = getattr(args, "data_root", "data_folder")
    batch_size   = getattr(args, "batch_size", 128)
    num_workers  = getattr(args, "num_workers", 0)
    pin_memory   = torch.cuda.is_available()
    test_frac    = float(getattr(args, "test_frac", 0.2))

    # ---- Load (flat) arrays
    X, y = load_dataset(dataset_name, root=data_root)  # X: [N, D] float32, y: [N] int64
    N, D = X.shape

    # Known image shapes
    mnist_like = {
        "mnist_basic",
        "mnist_rotated",        
        "mnist_background_images",
        "mnist_background_random",
        "fashion_mnist",
    }

    # ---- Stratified train/test split
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_test = max(1, int(round(len(c_idx) * test_frac)))
        test_idx.append(c_idx[:n_test])
        train_idx.append(c_idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    # Optional global shuffle (keeps stratification already applied)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ---- Convert to tensors (reshape if needed)
    xtr = torch.from_numpy(X_train)            # [Ntr, D]
    xte = torch.from_numpy(X_test)             # [Nte, D]
    input_size = int(D)

    ytr = torch.from_numpy(y_train)
    yte = torch.from_numpy(y_test)

    train_set = TensorDataset(xtr, ytr)
    test_set  = TensorDataset(xte, yte)

    # ---- DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, test_loader, input_size, dataset_name


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build loaders & detect input size
    train_loader, test_loader, input_size, dataset_name = build_dataloaders(args)

    # Build the model
    hidden_size = 500
    latent_size = args.latent_size

    if args.model == "VAE":
        model = VAE(input_size, hidden_size, latent_size).to(device)
    elif args.model == "MVAE":
        model = MVAE(input_size, hidden_size, latent_size).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    curves_dir = os.path.join(results_dir, "curves")

    num_epochs = args.epochs
    train_elbo_curve, test_elbo_curve = [], []
    train_msepp_curve, test_msepp_curve = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_data = batch_x.to(device).view(-1, input_size)
            batch_mean, batch_logvar, reconst_batch = model(batch_data)
            aver_loss = -model.compute_loss(batch_data, reconst_batch, batch_mean, batch_logvar)

            optimizer.zero_grad()
            aver_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs}, "
                    f"Batch {batch_idx + 1}, "
                    f"Aver_Loss: {aver_loss.item():.2f}"
                )

        model.eval()
        with torch.no_grad():
            train_epoch = gather_elbo_mse(model, train_loader, device, input_size)
            train_elbo_epoch = train_epoch["elbo"]
            train_elbo_curve.append(train_elbo_epoch)
            train_msepp_epoch = train_epoch["mse_per_pixel"]
            train_msepp_curve.append(train_msepp_epoch)

            test_epoch  = gather_elbo_mse(model, test_loader,  device, input_size)
            test_elbo_epoch = test_epoch["elbo"]
            test_elbo_curve.append(test_elbo_epoch)
            test_msepp_epoch = test_epoch["mse_per_pixel"]
            test_msepp_curve.append(test_msepp_epoch)

    os.makedirs(args.params_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(args.params_dir, f"{dataset_name}_zdim{latent_size}_{args.model}.pkl"),
    )

    save_curves(
        curves_dir,
        dataset=dataset_name,
        model=args.model,
        latent=latent_size,
        seed=args.seed,
        train_elbo_hist=train_msepp_curve,  # ELBO
        test_elbo_hist=test_msepp_curve,    # ELBO
        train_mse_hist=train_elbo_curve,    # MSE-per-pixel
        test_mse_hist=test_elbo_curve       # MSE-per-pixel
    )
    
    if args.evaluate:
        _ = evaluate_model(
            model=model,
            latent=latent_size,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            input_size=input_size,
            results_dir=results_dir,
            model_name=args.model.lower(),
            dataset_name=dataset_name,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
