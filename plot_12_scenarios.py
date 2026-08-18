import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize_scalar

# ==============================================================================
# Theoretical Quantum Bounds
# ==============================================================================

def helstrom_bound(alpha, eta=1.0, pi0=0.5):
    """Ideal Quantum Helstrom minimum error bound for BPSK"""
    pi1 = 1.0 - pi0
    overlap_sq = np.exp(-4.0 * eta * alpha**2)
    return 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - 4.0 * pi0 * pi1 * overlap_sq, 0.0)))

def get_optimal_kennedy_displacement(alpha, eta=1.0, nu=0.0, pi0=0.5, T=1.0):
    pi1 = 1.0 - pi0
    def loss(b):
        r0 = (eta * (alpha - b)**2 + nu) * T
        r1 = (eta * (alpha + b)**2 + nu) * T
        err0 = 1.0 - np.exp(-r0)
        err1 = np.exp(-r1)
        return pi0 * err0 + pi1 * err1
    res = minimize_scalar(loss, bounds=(alpha, alpha + 3.0), method='bounded')
    return res.x

def optimized_kennedy_bound(alpha, eta=1.0, nu=0.0, pi0=0.5, T=1.0):
    """Ideal Displacement-Optimized Kennedy bound"""
    pi1 = 1.0 - pi0
    if np.isscalar(alpha):
        b_opt = get_optimal_kennedy_displacement(alpha, eta=eta, nu=nu, pi0=pi0, T=T)
        r0 = (eta * (alpha - b_opt)**2 + nu) * T
        r1 = (eta * (alpha + b_opt)**2 + nu) * T
        return pi0 * (1.0 - np.exp(-r0)) + pi1 * np.exp(-r1)
    else:
        return np.array([optimized_kennedy_bound(a, eta=eta, nu=nu, pi0=pi0, T=T) for a in alpha])

# ==============================================================================
# Main Plotting Function
# ==============================================================================

def plot_12_scenarios(data_source="runs", output_prefix="quantum_receiver_12_scenarios"):
    """
    Plots a 3x4 multi-panel grid comparing all 12 physical scenarios.
    Accepts either a directory of runs ('runs/') or a combined CSV/Parquet file.
    """
    # 1. Load Data
    if os.path.isdir(data_source):
        csv_files = sorted(glob.glob(os.path.join(data_source, "*", "results.csv")))
        if not csv_files:
            print(f"No results.csv files found in '{data_source}/*/'")
            return
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    elif os.path.isfile(data_source):
        if data_source.endswith(".parquet"):
            df = pd.read_parquet(data_source)
        else:
            df = pd.read_csv(data_source)
    else:
        print(f"Error: Data source '{data_source}' not found.")
        return

    # Extract unique scenarios in order
    scenarios = df["scenario"].unique()
    n_scenarios = len(scenarios)
    print(f"Found {n_scenarios} scenarios: {list(scenarios)}")

    # 2. Setup Plot Grid (3 rows x 4 columns)
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    alpha_fine = np.linspace(0.05, 1.8, 200)
    ideal_helstrom = helstrom_bound(alpha_fine, eta=1.0)
    ideal_kennedy = optimized_kennedy_bound(alpha_fine, eta=1.0, nu=0.0)

    for idx, sc_name in enumerate(scenarios):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sc_df = df[df["scenario"] == sc_name]

        # Extract subsets
        dolinar_df = sc_df[sc_df["receiver"] == "Dolinar"].sort_values("alpha")
        kennedy_df = sc_df[sc_df["receiver"] == "Opt_Kennedy"].sort_values("alpha")

        # Theoretical Ideal Reference Curves
        ax.semilogy(alpha_fine, ideal_helstrom, 'k-', linewidth=2.0, alpha=0.85, label=r'Ideal Helstrom ($\eta=1.0$)')
        ax.semilogy(alpha_fine, ideal_kennedy, color='darkorange', linestyle=':', linewidth=2.0, label=r'Ideal Opt-Kennedy ($\eta=1.0$)')

        # Simulated Data Points
        if not dolinar_df.empty:
            ax.semilogy(dolinar_df["alpha"], dolinar_df["ber"], 'g^', markersize=6.5, label='Dolinar Receiver')
        if not kennedy_df.empty:
            ax.semilogy(kennedy_df["alpha"], kennedy_df["ber"], 'rx', markersize=6.5, markeredgewidth=1.6, label='Optimized Kennedy')

        # Clean Title & Formatting
        formatted_title = sc_name.replace("_", " ")
        ax.set_title(formatted_title, fontsize=11, fontweight='bold')
        ax.set_xlim(0.1, 1.8)
        ax.set_ylim(1e-7, 0.8)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        # Labels for outer subplots
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel(r'Amplitude $\alpha$', fontsize=10.5)
        if idx % ncols == 0:
            ax.set_ylabel('Bit Error Rate (BER)', fontsize=10.5)

        if idx == 0:
            ax.legend(loc='lower left', fontsize=8.5)

    # Hide any unused subplots if fewer than 12
    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Quantum Receiver BPSK Discrimination: 12 HPC Physical Scenarios Benchmark', fontsize=15, y=0.995)
    plt.tight_layout()

    # Save outputs
    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plots saved successfully:\n  • {png_path} (300 DPI)\n  • {pdf_path} (Vector PDF)")
    plt.close()

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "runs"
    plot_12_scenarios(data_source=src)
