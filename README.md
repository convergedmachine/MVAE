# Variational Autoencoder (VAE) and Multivariate-VAE (MVAE) Training & Evaluation

This repository provides a complete and reproducible PyTorch implementation for training, evaluating, and benchmarking **Variational Autoencoders (VAE)** and **Multivariate-Variational Autoencoders (MVAE)** on **MNIST** and **Fashion-MNIST** datasets.  
It includes integrated **ELBO & MSE tracking**, **logistic regression probes**, **clustering metrics**, and **curve export** in JSON/CSV/NPZ formats.

---

## 🌟 Features

- **Unified Training Script** (`train.py`):
  - Supports `VAE` and `MVAE` models
  - Works with `MNIST` and `Fashion-MNIST`
- **Evaluation Metrics**:
  - Reconstruction loss (MSE-per-pixel)
  - Evidence Lower Bound (ELBO)
  - Latent-space quality via:
    - Linear probe accuracy / NLL / Brier / ECE
    - Clustering quality (NMI, ARI)
- **Reproducibility**:
  - Fixed seeds and saved curves
  - JSON/CSV metric logging for analysis
- **Optimized performance**:
  - [Intel® Extension for Scikit-learn](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html) acceleration (`sklearnex`)
- **Fully self-contained pipeline** — no external configs required.

---

## 🧩 Project Structure

```

.
├── train.py                # Main training and evaluation script
├── models.py               # Defines VAE and MVAE architectures
├── utils.py                # Visualization and curve utilities
├── results/
│   ├── curves/             # JSON/CSV/NPZ training curves
│   └── metrics/            # Evaluation summaries
└── trained_parameters/     # Saved PyTorch model weights

````

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision numpy scikit-learn scikit-learn-intelex matplotlib
````

> ✅ *Optional:* For Intel CPUs, `scikit-learn-intelex` accelerates probe and clustering computations automatically.

---

## 🚀 Usage

### 1️⃣ Train and Evaluate a Model

```bash
python train.py \
  --dataset mnist \
  --model VAE \
  --latent-size 32 \
  --epochs 200 \
  --batch-size 100 \
  --lr 1e-3 \
  --seed 777
```

### 2️⃣ Example for MVAE on Fashion-MNIST

```bash
python train.py \
  --dataset fashion_mnist \
  --model MVAE \
  --latent-size 20 \
  --epochs 150 \
  --batch-size 128
```

---

## 📈 Output Artifacts

After training, the following are produced:

| Artifact                                 | Description                     |
| ---------------------------------------- | ------------------------------- |
| `results/<dataset>/<model>_metrics.json` | Final evaluation metrics        |
| `results/<dataset>/curves/*.json`        | Per-epoch ELBO and MSE curves   |
| `trained_parameters/*.pkl`               | Saved model weights             |
| `logs/*.out` *(optional)*                | Console logs (when using nohup) |

Example metric summary:

```json
{
  "model": "mvae",
  "recon/mse_per_pixel": float,
  "elbo/train": float,
  "probe/acc": float,
  "cluster/nmi": float,
  "eff/n_params": int
}
```

---

## 📊 Evaluation Metrics Explained

| Metric                                 | Type           | Goal  | Description                                                        |
| :------------------------------------- | :------------- | :---- | :----------------------------------------------------------------- |
| **ELBO**                               | Generative     | ↑     | Evidence Lower Bound (training objective)                          |
| **MSE-per-pixel**                      | Reconstruction | ↓     | Average pixel-wise reconstruction error                            |
| **Probe Accuracy / NLL / Brier / ECE** | Latent quality | ↑ / ↓ | Linear classifier evaluation of latent structure                   |
| **NMI / ARI**                          | Clustering     | ↑     | Unsupervised alignment between cluster assignments and true labels |
| **#Params / Probe Train Sec**          | Efficiency     | —     | Model size and training time of the probe                          |

---

## 🧠 Implementation Highlights

* Uses `AdamW` optimizer for stable training
* Automatically computes and stores ELBO & MSE each epoch
* Uses **μ** (mean vector) as latent representation for downstream probes
* Fully vectorized computation for `KMeans`, `LogisticRegression`, and `ECE`
* Compatible with GPU (`cuda`) or CPU

---

## 🧩 License

This repository is released under the **MIT License**.

---

## 🙌 Acknowledgments

* **D. P. Kingma and M. Welling** – *Auto-Encoding Variational Bayes (ICLR 2014)*
* **Intel® Scikit-Learn-Extension** for accelerated evaluation
* The **PyTorch** and **TorchVision** teams for dataset loaders and model utilities

---

**Author:** Mehmet Can Yavuz, PhD
**Affiliation:** Işık University – Multimedia Lab

---

## 📄 License

MIT License.
See `LICENSE` for details.
