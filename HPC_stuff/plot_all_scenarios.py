import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.special as sp
from scipy.optimize import minimize_scalar

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

def plot_all_scenarios(data_source="runs", output_prefix="quantum_receiver_scenarios"):
    """
    Dynamically plots ALL scenarios found in the dataset.
    Generates:
      1. A multi-page PDF (12 subplots per page) for high-resolution reading.
      2. A full comprehensive master poster image (all scenarios in one giant grid).
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

    scenarios = df["scenario"].unique()
    n_scenarios = len(scenarios)
    print(f"Found {n_scenarios} total scenarios in dataset.")

    alpha_fine = np.linspace(0.05, 1.8, 200)
    ideal_helstrom = helstrom_bound(alpha_fine, eta=1.0)
    ideal_kennedy = optimized_kennedy_bound(alpha_fine, eta=1.0, nu=0.0)

    plots_per_page = 12
    n_pages = math.ceil(n_scenarios / plots_per_page)
    pdf_path = f"{output_prefix}_multipage.pdf"

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(n_pages):
            start_sc = page_idx * plots_per_page
            end_sc = min(start_sc + plots_per_page, n_scenarios)
            page_scenarios = scenarios[start_sc:end_sc]

            fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sc_name in enumerate(page_scenarios):
                ax = axes[i]
                sc_df = df[df["scenario"] == sc_name]
                dolinar_df = sc_df[sc_df["receiver"] == "Dolinar"].sort_values("alpha")
                kennedy_df = sc_df[sc_df["receiver"] == "Opt_Kennedy"].sort_values("alpha")

                ax.semilogy(alpha_fine, ideal_helstrom, 'k-', linewidth=2.0, alpha=0.85, label=r'Ideal Helstrom ($\eta=1.0$)')
                ax.semilogy(alpha_fine, ideal_kennedy, color='darkorange', linestyle=':', linewidth=2.0, label=r'Ideal Opt-Kennedy ($\eta=1.0$)')

                if not dolinar_df.empty:
                    ax.semilogy(dolinar_df["alpha"], dolinar_df["ber"], 'g^', markersize=6.5, label='Dolinar Receiver')
                if not kennedy_df.empty:
                    ax.semilogy(kennedy_df["alpha"], kennedy_df["ber"], 'rx', markersize=6.5, markeredgewidth=1.6, label='Optimized Kennedy')

                ax.set_title(sc_name.replace("_", " "), fontsize=11, fontweight='bold')
                ax.set_xlim(0.1, 1.8)
                ax.set_ylim(1e-7, 0.8)
                ax.grid(True, which='both', linestyle=':', alpha=0.5)

                if i >= 8:
                    ax.set_xlabel(r'Amplitude $\alpha$', fontsize=10.5)
                if i % 4 == 0:
                    ax.set_ylabel('Bit Error Rate (BER)', fontsize=10.5)
                if i == 0:
                    ax.legend(loc='lower left', fontsize=8.5)

            for i in range(len(page_scenarios), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f'Quantum Receiver Benchmark (Scenarios {start_sc + 1} to {end_sc} of {n_scenarios})', fontsize=15, y=0.995)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    ncols = 6
    nrows = math.ceil(n_scenarios / ncols)
    fig_w = ncols * 4.5
    fig_h = nrows * 3.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for idx, sc_name in enumerate(scenarios):
        ax = axes[idx]
        sc_df = df[df["scenario"] == sc_name]
        dolinar_df = sc_df[sc_df["receiver"] == "Dolinar"].sort_values("alpha")
        kennedy_df = sc_df[sc_df["receiver"] == "Opt_Kennedy"].sort_values("alpha")

        ax.semilogy(alpha_fine, ideal_helstrom, 'k-', linewidth=1.8, alpha=0.85, label='Ideal Helstrom')
        ax.semilogy(alpha_fine, ideal_kennedy, color='darkorange', linestyle=':', linewidth=1.8, label='Ideal Opt-Kennedy')

        if not dolinar_df.empty:
            ax.semilogy(dolinar_df["alpha"], dolinar_df["ber"], 'g^', markersize=5.5, label='Dolinar')
        if not kennedy_df.empty:
            ax.semilogy(kennedy_df["alpha"], kennedy_df["ber"], 'rx', markersize=5.5, markeredgewidth=1.4, label='Opt-Kennedy')

        ax.set_title(sc_name.replace("_", " "), fontsize=9.5, fontweight='bold')
        ax.set_xlim(0.1, 1.8)
        ax.set_ylim(1e-7, 0.8)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel(r'Amplitude $\alpha$', fontsize=9.5)
        if idx % ncols == 0:
            ax.set_ylabel('BER', fontsize=9.5)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=7.5)

    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Complete Quantum Receiver Sweep ({n_scenarios} Physical Scenarios)', fontsize=16, y=0.998)
    plt.tight_layout()

    poster_png = f"{output_prefix}_all_{n_scenarios}.png"
    plt.savefig(poster_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Generated successfully:")
    print(f"  • Multi-Page PDF (12 per page): {pdf_path}")
    print(f"  • Complete Master Poster ({nrows}x{ncols} grid): {poster_png}")

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "runs"
    plot_all_scenarios(data_source=src)
