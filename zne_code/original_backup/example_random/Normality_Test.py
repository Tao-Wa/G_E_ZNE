"""
Normality Test for Uniform-Path Noise Distribution

This script reads noise and Pauli weight data, computes self-normalized importance weights,
fits a lognormal distribution to the weighted noise, and generates diagnostic plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import lognorm

# ========================== Configuration ==========================
BASE_DIR = "weigh_data"
NOISE_FILE = os.path.join(BASE_DIR, "noise_random_10_1_0.txt")
PAULI_WEIGHT_FILE = os.path.join(BASE_DIR, "pauli_weight_random_10_1_0.txt")
SAVE_DIR = "log_norm_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========================== Data Loading ==========================
def load_array_from_file(filepath, var_name):
    """
    Load a Python array variable from a text file using exec().
    The file should contain a line like `var_name = [ ... ]`.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    namespace = {}
    exec(content, namespace)
    return np.asarray(namespace[var_name], dtype=float)

# Load data
noise = load_array_from_file(NOISE_FILE, 'noise')               # W_alpha
tilde_q = load_array_from_file(PAULI_WEIGHT_FILE, 'pauli_weight')  # unnormalized q

print(f"Number of samples: {len(noise)}")
assert len(noise) == len(tilde_q), "Arrays must have the same length"

# ========================== Importance Weights ==========================
# Numerically stable self-normalized importance weights
log_w = -np.log(tilde_q)
log_w -= np.max(log_w)          # stabilization (subtract max to avoid underflow)
w = np.exp(log_w)
w /= np.sum(w)                  # self-normalized

ess = 1.0 / np.sum(w**2)
print(f"Effective Sample Size (ESS) = {ess:.2f}")

# Keep only positive noise values (noise should be > 0 theoretically)
mask = noise > 0
noise_pos = noise[mask]
w_pos = w[mask]
w_pos /= np.sum(w_pos)          # renormalize

log_noise = np.log(noise_pos)

# ========================== Lognormal Fit (Weighted) ==========================
# Weighted mean and variance on log scale
mu = np.sum(w_pos * log_noise)
sigma = np.sqrt(np.sum(w_pos * (log_noise - mu)**2))

print(f"Fitted lognormal parameters: mu = {mu:.4f}, sigma = {sigma:.4f}")

# ========================== Plotting ==========================
def plot_weighted_histogram_and_fit(noise_pos, w_pos, mu, sigma, save_path=None):
    """Plot weighted histogram of noise with fitted lognormal PDF."""
    plt.figure(figsize=(6, 4))
    plt.hist(noise_pos, bins=200, weights=w_pos, density=True,
             alpha=0.5, label="Uniform-path weighted histogram")
    x = np.linspace(noise_pos.min(), noise_pos.max(), 600)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    plt.plot(x, pdf, 'r', lw=2, label="Fitted lognormal PDF")
    plt.xlabel("Noise W")
    plt.ylabel("Density (uniform paths)")
    plt.title("Uniform-path noise distribution & lognormal fit")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_log_noise_histogram_and_gaussian(log_noise, w_pos, mu, sigma, save_path=None):
    """Plot weighted histogram of log(noise) with fitted Gaussian PDF."""
    plt.figure(figsize=(6, 4))
    plt.hist(log_noise, bins=200, weights=w_pos, density=True,
             alpha=0.5, label="Weighted histogram of log(noise)")
    x = np.linspace(log_noise.min(), log_noise.max(), 600)
    gaussian_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    plt.plot(x, gaussian_pdf, 'r', lw=2, label="Gaussian fit")
    plt.xlabel("log(noise)")
    plt.ylabel("Density (uniform paths)")
    plt.title("log(noise) under uniform paths")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_weighted_qq(log_noise, w_pos, mu, sigma, save_path=None):
    """Generate weighted QQ plot of log(noise) against normal distribution."""
    # Sort by log_noise
    sorted_idx = np.argsort(log_noise)
    log_noise_sorted = log_noise[sorted_idx]
    w_sorted = w_pos[sorted_idx]
    cum_weights = np.cumsum(w_sorted)
    cum_weights /= cum_weights[-1]

    # Theoretical quantiles (normal)
    p = np.linspace(0.01, 0.99, 500)
    quantiles_theoretical = stats.norm.ppf(p, loc=mu, scale=sigma)

    # Weighted sample quantiles
    quantiles_sample = np.interp(p, cum_weights, log_noise_sorted)

    # Correlation coefficient
    corr = np.corrcoef(quantiles_theoretical, quantiles_sample)[0, 1]
    print(f"Weighted quantile correlation r = {corr:.4f}")

    # QQ plot
    plt.figure(figsize=(5, 5))
    plt.scatter(quantiles_theoretical, quantiles_sample, s=10, alpha=0.6)
    plt.plot([quantiles_theoretical.min(), quantiles_theoretical.max()],
             [quantiles_theoretical.min(), quantiles_theoretical.max()],
             'r-', lw=2, label="Identity line")
    plt.xlabel("Theoretical quantiles (normal)")
    plt.ylabel("Sample quantiles (weighted)")
    plt.title("Weighted QQ plot of log(noise)")
    plt.text(0.02, 0.98, f'r = {corr:.4f}',
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return corr

# Generate plots
plot_weighted_histogram_and_fit(noise_pos, w_pos, mu, sigma,
                                save_path=os.path.join(SAVE_DIR, "weighted_histogram_lognormal.png"))

plot_log_noise_histogram_and_gaussian(log_noise, w_pos, mu, sigma,
                                      save_path=os.path.join(SAVE_DIR, "log_noise_histogram_gaussian.png"))

correlation = plot_weighted_qq(log_noise, w_pos, mu, sigma,
                               save_path=os.path.join(SAVE_DIR, "qq_plot_log_noise.png"))

print("All plots have been generated and saved.")