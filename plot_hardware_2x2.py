import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize_scalar

# ==============================================================================
# Theoretical Quantum Bounds
# ==============================================================================

def helstrom_bound(alpha, eta=1.0, pi0=0.5):
    """Quantum Helstrom minimum error probability bound for BPSK"""
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
    """Displacement-Optimized Kennedy bound (beta = -beta_opt > alpha)"""
    pi1 = 1.0 - pi0
    if np.isscalar(alpha):
        b_opt = get_optimal_kennedy_displacement(alpha, eta=eta, nu=nu, pi0=pi0, T=T)
        r0 = (eta * (alpha - b_opt)**2 + nu) * T
        r1 = (eta * (alpha + b_opt)**2 + nu) * T
        return pi0 * (1.0 - np.exp(-r0)) + pi1 * np.exp(-r1)
    else:
        return np.array([optimized_kennedy_bound(a, eta=eta, nu=nu, pi0=pi0, T=T) for a in alpha])

# ==============================================================================
# Plot 2: 2x2 Grid - Lab vs. Commercial (SNSPD vs. APD)
# ==============================================================================

def plot_hardware_2x2(data_source="combined_results.csv", output_dir="images", output_filename="hardware_2x2_comparison.png"):
    """
    Plots a 2x2 multi-panel grid comparing:
      - Top-Left:  Lab State-of-the-Art SNSPD (NIST / JPL Record)
      - Top-Right: Commercial Turnkey SNSPD (Single Quantum Eos)
      - Bottom-Left:  Lab APD Benchmark (Becerra NIST 2013 / Excelitas)
      - Bottom-Right: Commercial InGaAs APD (ID Quantique ID230)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)
    # 1. Load Data
    if os.path.isdir(data_source):
        csv_files = sorted(glob.glob(os.path.join(data_source, "*", "results.csv")))
        if not csv_files:
            print(f"No CSV files found in '{data_source}'")
            return
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    elif os.path.isfile(data_source):
        df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        print(f"Error: Data source '{data_source}' not found.")
        return

    panels = [
        {
            "match": "NIST_JPL",
            "title": r"1. Lab State-of-the-Art SNSPD (NIST / JPL)",
            "subtitle": r"($\eta=0.98, \nu=10^{-5}, \tau=0.003T, \delta\phi=1.0^\circ$)"
        },
        {
            "match": "SingleQuantum",
            "title": r"2. Commercial Turnkey SNSPD (Single Quantum)",
            "subtitle": r"($\eta=0.92, \nu=2\times 10^{-4}, \tau=0.006T, \delta\phi=2.5^\circ$)"
        },
        {
            "match": "Becerra",
            "title": r"3. Lab APD Benchmark (Becerra et al., NIST 2013)",
            "subtitle": r"($\eta=0.55, \nu=1.6\times 10^{-3}, \tau=0.014T, \delta\phi=4.0^\circ$)"
        },
        {
            "match": "IDQuantique",
            "title": r"4. Commercial InGaAs APD (ID Quantique ID230)",
            "subtitle": r"($\eta=0.25, \nu=10^{-2}, \tau=0.020T, \delta\phi=8.0^\circ$)"
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), sharex=True, sharey=True)
    axes = axes.flatten()

    alpha_fine = np.linspace(0.05, 1.8, 300)
    ideal_helstrom = helstrom_bound(alpha_fine, eta=1.0)
    ideal_kennedy = optimized_kennedy_bound(alpha_fine, eta=1.0, nu=0.0)

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        
        matched_name = None
        for s in df["scenario"].unique():
            if panel["match"].lower() in s.lower():
                matched_name = s
                break
                
        if matched_name is None:
            matched_name = df["scenario"].unique()[min(idx, len(df["scenario"].unique()) - 1)]
            print(f"Panel {idx+1}: Could not find match for '{panel['match']}', using '{matched_name}'")
        else:
            print(f"Panel {idx+1}: Matched '{matched_name}'")

        sc_df = df[df["scenario"] == matched_name]
        dolinar_df = sc_df[sc_df["receiver"] == "Dolinar"].sort_values("alpha")
        kennedy_df = sc_df[sc_df["receiver"] == "Opt_Kennedy"].sort_values("alpha")

        # 1. Plot Ideal Reference Limits (eta=1.0)
        ax.semilogy(alpha_fine, ideal_helstrom, 'k-', linewidth=2.8, alpha=0.9, label=r'Ideal Helstrom Limit ($\eta=1.0$)')
        ax.semilogy(alpha_fine, ideal_kennedy, color='darkorange', linestyle=':', linewidth=2.6, label=r'Ideal Opt-Kennedy Bound ($\eta=1.0$)')

        # 2. Plot Real Hardware Simulations
        if not dolinar_df.empty:
            ax.semilogy(dolinar_df["alpha"], dolinar_df["ber"], 'g^', markersize=8.5, label='Dolinar Receiver Simulation')
        if not kennedy_df.empty:
            ax.semilogy(kennedy_df["alpha"], kennedy_df["ber"], 'rx', markersize=8.5, markeredgewidth=2.0, label='Optimized Kennedy Simulation')

        # Titles and Formatting
        ax.set_title(f"{panel['title']}\n{panel['subtitle']}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0.1, 1.8)
        ax.set_ylim(1e-7, 0.8)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        # Exact Matched Ticks from plot_ideal_benchmark
        ax.tick_params(axis='both', which='major', labelsize=13.5, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=12, length=3.5)

        if idx >= 2:
            ax.set_xlabel(r'Coherent State Amplitude $\alpha$', fontsize=16, labelpad=10)
        if idx % 2 == 0:
            ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, labelpad=10)

        # Legend on bottom-left of each panel
        ax.legend(loc='lower left', fontsize=13.5, framealpha=0.92, edgecolor='gray')

    plt.suptitle('Quantum Receiver Discrimination: Real-World Hardware Comparison (Lab vs. Commercial)', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save PNG and PDF
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    pdf_out = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_out, bbox_inches='tight')
    print(f"Saved 2x2 Hardware Comparison to:\n  • {output_file}\n  • {pdf_out}")
    plt.show()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "combined_results.csv"
    plot_hardware_2x2(data_source=src, output_dir="images")
