import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, data_type="binary"):
        super(VAE, self).__init__()
        # Encoder: layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        # Decoder: layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc41 = nn.Linear(hidden_size, input_size)
        self.fc42 = nn.Linear(hidden_size, input_size)
        # data_type: can be "binary" or "real"
        self.data_type = data_type

    def compute_loss(self, x, reconst_x, mean, log_var):
        # ELBO(Evidence Lower Bound) is the objective of VAE, we train the model just to maximize the ELBO.
        
        reconst_error = -torch.nn.functional.binary_cross_entropy(reconst_x, x, reduction='sum')
        # see Appendix B from VAE paper: "Kingma and Welling. Auto-Encoding Variational Bayes. ICLR-2014."
        # -KL[q(z|x)||p(z)] = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        elbo = (reconst_error - kl_divergence) / len(x)
        return elbo    
    
    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        mean, log_var = self.fc21(h1), self.fc22(h1)
        return mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        mu, sigma = mean, torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def decode(self, z):
        h3 = torch.tanh(self.fc3(z))
        if self.data_type == "real":
            mean, log_var = torch.sigmoid(self.fc41(h3)), self.fc42(h3)
            return mean, log_var
        else:
            logits = self.fc41(h3)
            probs = torch.sigmoid(logits)
            return probs

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z_mean, z_logvar, self.decode(z)


class MVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, data_type="binary"):
        super(MVAE, self).__init__()
        self.latent_size = latent_size

        # ----- Encoder -----
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)  # mean head (pre-coupling)
        self.fc22 = nn.Linear(hidden_size, latent_size)  # log-variance head (per-sample)
        # Global coupling matrix C (shared across the dataset)
        self.coupling = nn.Linear(latent_size, latent_size, bias=False)

        # ----- Decoder -----
        self.fc3  = nn.Linear(latent_size, hidden_size)
        self.fc41 = nn.Linear(hidden_size, input_size)   # logits/mean
        self.fc42 = nn.Linear(hidden_size, input_size)   # log_var for real-valued output

        # data_type: "binary" → Bernoulli (BCE), "real" → Gaussian (MSE with learned variance)
        self.data_type = data_type

    # ----------- Loss terms (ELBO = recon_loglik - KL) -----------       
    def compute_loss(self, x, reconst_x, mean, L) -> torch.Tensor:
        B, d = mean.shape
        if L.dim() == 2:
            # If a shared L was passed, broadcast to batch
            L = L.unsqueeze(0).expand(B, -1, -1)
        reconst_error = -torch.nn.functional.binary_cross_entropy(reconst_x, x, reduction='sum')

        # --- KL(q(z|x) || p(z)) for q = N(mean, Sigma=L L^T), p=N(0,I)
        # trace(Sigma) = ||L||_F^2 per-sample (since tr(LL^T) = sum of squares of L entries)
        trace_Sigma = (L ** 2).sum(dim=(1, 2))  # [B]
        mean_norm2  = (mean ** 2).sum(dim=1)    # [B]
        # log|Sigma| = 2 * log|L|   (since Sigma = L L^T)
        # Use slogdet on the batch of L (no need for Sigma or Cholesky)
        sign, logabsdet_L = torch.linalg.slogdet(L)      # each [B]
        # (If sign <= 0 arises during early training, gradients will push to a valid region;
        # adding small jitter to L can help numerically if needed.)
        logdet_Sigma = 2.0 * logabsdet_L                 # [B]

        kl_divergence = -0.5 * torch.sum(d + logdet_Sigma - trace_Sigma - mean_norm2)  # [B]
        elbo = (reconst_error - kl_divergence) / len(x)
        return elbo

    # ----------- Encoder -----------
    def encode(self, x):
        """
        Returns:
          mean: [B, d]
          L   : [B, d, d] with L_i = C @ diag(sigma_i), sigma_i = exp(0.5 * logvar_i)
        """
        h1 = torch.tanh(self.fc1(x))
        mu = self.fc21(h1)                     # [B, d]  (pre-coupling mean)
        logvar = self.fc22(h1)                 # [B, d]
        sigma = torch.exp(0.5 * logvar)        # [B, d], strictly positive

        # Global coupling C (learned); do NOT detach—gradients should flow into C
        C = self.coupling.weight               # [d, d]

        # mean after coupling (you can choose to couple the mean or keep uncoupled; here we couple)
        mean = self.coupling(mu)               # [B, d]

        # Build per-sample L = C @ diag(sigma)
        # diag_embed(sigma): [B, d, d]; C.unsqueeze(0): [1, d, d]
        L = torch.matmul(C.unsqueeze(0), torch.diag_embed(sigma))  # [B, d, d]
        return mean, L

    # ----------- Reparameterization -----------
    @staticmethod
    def reparameterize(mean, L):
        """
        z = mean + L @ eps,   eps ~ N(0, I)
        Inputs:
          mean: [B, d]
          L   : [B, d, d]  (or [d, d] to be broadcast)
        Output:
          z   : [B, d]
        """
        B, d = mean.shape
        if L.dim() == 2:
            L = L.unsqueeze(0).expand(B, -1, -1)
        eps = torch.randn(B, d, device=mean.device, dtype=mean.dtype)  # [B, d]
        z = mean + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)      # [B, d]
        return z

    # ----------- Decoder -----------
    def decode(self, z):
        h3 = torch.tanh(self.fc3(z))
        if self.data_type == "real":
            mean = self.fc41(h3)                 # recon mean
            log_var = self.fc42(h3)              # recon log-var (per-dim)
            return mean, log_var
        else:
            logits = self.fc41(h3)
            probs = torch.sigmoid(logits)        # Bernoulli probs
            return probs

    # ----------- Forward -----------
    def forward(self, x):
        z_mean, z_L = self.encode(x)       # [B, d], [B, d, d]
        z = self.reparameterize(z_mean, z_L)
        recon = self.decode(z)
        return z_mean, z_L, recon
