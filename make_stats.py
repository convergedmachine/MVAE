import numpy as np
import torch
from torchvision import transforms
from train import load_dataset

# All datasets to check
DATASETS = [
    "mnist_basic",
    "mnist_rotated",
    "mnist_background_images",
    "mnist_background_random",
    "fashion_mnist",
    "cifar10",
    "cifar100",
    "freyface",
]

def check_dataset(name, root="data_folder"):
    X, y = load_dataset(name, root=root)
    Xmin, Xmax = np.min(X), np.max(X)

    # Heuristic: binary if only 0 and 1 (or close)
    unique_vals = np.unique(X)
    is_binary = np.all(np.isin(unique_vals, [0.0, 1.0])) or (
        np.isclose(Xmin, 0.0, atol=1e-4) and np.isclose(Xmax, 1.0, atol=1e-4)
    )

    print(f"\n=== {name.upper()} ===")
    print(f"Shape: {X.shape}, dtype: {X.dtype}")
    print(f"Range: min={Xmin:.4f}, max={Xmax:.4f}")
    print(f"Binary: {is_binary}")
    print(f"Unique values (sample): {unique_vals[:10]}")
    print(f"Num unique: {len(unique_vals)}")

if __name__ == "__main__":
    for ds in DATASETS:
        try:
            check_dataset(ds)
        except Exception as e:
            print(f"\n[!] Failed to load {ds}: {e}")
