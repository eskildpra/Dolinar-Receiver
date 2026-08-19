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

def homodyne_bound(alpha, pi0=0.5):
    """Standard Quantum Limit (SQL) via ideal homodyne detection"""
    pi1 = 1.0 - pi0
    x_th = (1.0 / (4.0 * np.maximum(alpha, 1e-6))) * np.log(pi1 / pi0)
    err0 = 0.5 * sp.erfc(np.sqrt(2.0) * (alpha - x_th))
    err1 = 0.5 * sp.erfc(np.sqrt(2.0) * (alpha + x_th))
    return pi0 * err0 + pi1 * err1

def standard_kennedy_bound(alpha, eta=1.0, pi0=0.5):
    """Standard Kennedy bound (fixed nulling beta = -alpha)"""
    pi1 = 1.0 - pi0
    return pi1 * np.exp(-4.0 * eta * alpha**2)

def get_optimal_kennedy_displacement(alpha, eta=1.0, nu=0.0, pi0=0.5, T=1.0):
    """Calculates optimal static over-displacement magnitude beta_opt for Kennedy receiver"""
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
# Plotting from Combined Results
# ==============================================================================

def plot_ideal_benchmark(data_source="combined_results.csv", scenario_name="01_Ideal", output_file="ideal_benchmark_comparison.png"):
    """
    Loads simulated data from data_source and plots it against all 4 fundamental theoretical bounds.
    """
    # 1. Load Data
    if os.path.isdir(data_source):
        csv_files = sorted(glob.glob(os.path.join(data_source, "*", "results.csv")))
        if not csv_files:
            print(f"No CSV files found in {data_source}")
            return
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    elif os.path.isfile(data_source):
        if data_source.endswith(".parquet"):
            df = pd.read_parquet(data_source)
        else:
            df = pd.read_csv(data_source)
    else:
        print(f"Error: Could not find '{data_source}'.")
        return

    # Find the ideal scenario (or first scenario)
    available_scenarios = df["scenario"].unique()
    match_sc = None
    for s in available_scenarios:
        if scenario_name.lower() in s.lower():
            match_sc = s
            break
    if match_sc is None:
        match_sc = available_scenarios[0]
        print(f"Scenario '{scenario_name}' not found. Using '{match_sc}' instead.")
    else:
        print(f"Plotting scenario: '{match_sc}'")

    sc_df = df[df["scenario"] == match_sc]
    dolinar_df = sc_df[sc_df["receiver"] == "Dolinar"].sort_values("alpha")
    kennedy_df = sc_df[sc_df["receiver"] == "Opt_Kennedy"].sort_values("alpha")

    # 2. Compute Theoretical Bounds
    alpha_grid = np.linspace(0.05, 1.8, 300)
    helstrom_curve = helstrom_bound(alpha_grid, eta=1.0, pi0=0.5)
    homodyne_curve = homodyne_bound(alpha_grid, pi0=0.5)
    std_kennedy_curve = standard_kennedy_bound(alpha_grid, eta=1.0, pi0=0.5)
    opt_kennedy_curve = optimized_kennedy_bound(alpha_grid, eta=1.0, nu=0.0, pi0=0.5)

    # 3. Create the Benchmark Plot with Larger Fonts & Bottom-Left Legend
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Continuous Quantum & Classical Limits
    ax.semilogy(alpha_grid, helstrom_curve, 'k-', linewidth=2.8, label='Helstrom Bound (Quantum Limit)')
    ax.semilogy(alpha_grid, homodyne_curve, 'g--', linewidth=2.4, label='Homodyne Detection (Standard Quantum Limit - SQL)')
    ax.semilogy(alpha_grid, std_kennedy_curve, 'c-.', linewidth=2.4, label=r'Standard Kennedy Bound ($\beta=-\alpha$)')
    ax.semilogy(alpha_grid, opt_kennedy_curve, color='darkorange', linestyle=':', linewidth=2.6, label=r'Optimized Kennedy Bound ($\beta=-\beta_{opt}$)')

    # Simulated Data Points from HPC Run
    if not dolinar_df.empty:
        ax.semilogy(dolinar_df["alpha"], dolinar_df["ber"], 'g^', markersize=8.5, label=r'Dolinar Receiver Simulation (Geremia HJB)')
    if not kennedy_df.empty:
        ax.semilogy(kennedy_df["alpha"], kennedy_df["ber"], 'rx', markersize=8.5, markeredgewidth=2.0, label=r'Optimized Kennedy Simulation ($\beta=-\beta_{opt}$)')

    # Typography & Sizing
    ax.set_title('BPSK Quantum Receiver Discrimination: Benchmark against Ideal Theoretical Limits', fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel(r'Coherent State Amplitude $\alpha$ (Mean Photon Number $\bar{n}=\alpha^2$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=14, labelpad=10)
    
    ax.tick_params(axis='both', which='major', labelsize=12.5, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=11, length=3.5)
    
    ax.set_ylim(1e-7, 0.8)
    ax.set_xlim(0.1, 1.8)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Legend Placed at Bottom-Left with Increased Font Size
    ax.legend(loc='lower left', fontsize=12, framealpha=0.92, edgecolor='gray')
    
    plt.tight_layout()

    # Save PNG and PDF
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    pdf_out = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_out, bbox_inches='tight')
    print(f"Saved benchmark figure to:\n  • {output_file}\n  • {pdf_out}")
    plt.show()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "combined_results.csv"
    sc_target = sys.argv[2] if len(sys.argv) > 2 else "01_Ideal"
    plot_ideal_benchmark(data_source=src, scenario_name=sc_target)
