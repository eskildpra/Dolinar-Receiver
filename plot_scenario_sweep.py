import os
import glob
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
# Generalized Scenario Sweep Plotter
# ==============================================================================

def plot_scenario_sweep(data_source="combined_results.csv", 
                        filter_keyword=None, 
                        scenario_ids=None,
                        show_kennedy=False,
                        show_theory_curves=True,
                        title=None,
                        output_file=None):
    """
    Plots a multi-curve sweep of ANY selected physical scenarios on a single figure.
    
    Parameters:
      data_source: path to combined_results.csv, parquet, or runs/ directory
      filter_keyword: filter scenarios by substring (e.g. 'Efficiency', 'Phase_Jitter', 'Latency', 'Dark_Counts')
      scenario_ids: list of integer IDs (e.g. [1, 2, 3, 4, 5, 6])
      show_kennedy: if True, also plots the Optimized Kennedy receiver data points for each scenario
      show_theory_curves: if True, plots dashed theoretical Helstrom bounds per scenario
      title: custom title for the plot
      output_file: output image file name
    """
    # 1. Load Data
    if os.path.isdir(data_source):
        csv_files = sorted(glob.glob(os.path.join(data_source, "*", "results.csv")))
        if not csv_files:
            print(f"No results.csv files found in '{data_source}'")
            return
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    elif os.path.isfile(data_source):
        df = pd.read_parquet(data_source) if data_source.endswith(".parquet") else pd.read_csv(data_source)
    else:
        print(f"Error: Data source '{data_source}' not found.")
        return

    all_scenarios = list(df["scenario"].unique())

    # 2. Filter Scenarios
    selected_scenarios = []
    if scenario_ids is not None and len(scenario_ids) > 0:
        for sc in all_scenarios:
            for s_id in scenario_ids:
                if sc.startswith(f"{int(s_id):02d}_") or sc.startswith(f"{int(s_id)}_"):
                    selected_scenarios.append(sc)
    elif filter_keyword is not None and filter_keyword.strip() != "":
        kw = filter_keyword.lower()
        selected_scenarios = [s for s in all_scenarios if kw in s.lower()]
    else:
        # Default: if keyword 'efficiency' exists use it, otherwise use all
        eff = [s for s in all_scenarios if "efficiency" in s.lower()]
        selected_scenarios = eff if eff else all_scenarios

    if not selected_scenarios:
        print(f"No scenarios matched filter criteria (filter='{filter_keyword}', ids={scenario_ids}).")
        print(f"Available scenarios: {all_scenarios}")
        return

    print(f"Plotting {len(selected_scenarios)} scenarios: {selected_scenarios}")

    # Auto-generate Title and Output Name if not provided
    if title is None:
        if filter_keyword:
            clean_kw = filter_keyword.replace("_", " ").title()
            title = f'Impact of {clean_kw} on Quantum Receiver Performance'
        else:
            title = 'Quantum Receiver Multi-Scenario Parameter Sweep'

    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    if output_file is None:
        tag = filter_keyword.lower().replace(" ", "_") if filter_keyword else "custom_sweep"
        output_file = os.path.join(output_dir, f"{tag}_comparison.png")
    elif not os.path.isabs(output_file) and not output_file.startswith("images"):
        output_file = os.path.join(output_dir, output_file)

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    alpha_fine = np.linspace(0.05, 1.8, 300)

    # Master Ideal Helstrom Limit (eta=1.0)
    ideal_helstrom = helstrom_bound(alpha_fine, eta=1.0)
    ax.semilogy(alpha_fine, ideal_helstrom, 'k-', linewidth=3.0, label=r'Ideal Helstrom Limit ($\eta=1.00$)', zorder=10)

    # Colormap across scenarios
    n_sc = len(selected_scenarios)
    colors = cm.plasma(np.linspace(0.05, 0.90, n_sc)) if "latency" in title.lower() or "jitter" in title.lower() else cm.viridis(np.linspace(0.1, 0.9, n_sc))

    for idx, sc_name in enumerate(selected_scenarios):
        sc_df = df[df["scenario"] == sc_name]
        dolinar_df = sc_df[sc_df["receiver"] == "Dolinar"].sort_values("alpha")
        kennedy_df = sc_df[sc_df["receiver"] == "Opt_Kennedy"].sort_values("alpha")
        
        eta_val = sc_df["eta"].iloc[0] if "eta" in sc_df.columns else 1.0
        label_name = sc_name.replace("_", " ")

        # Optional theoretical Helstrom curve for efficiency
        if show_theory_curves and eta_val < 1.0 and ("efficiency" in sc_name.lower()):
            th_curve = helstrom_bound(alpha_fine, eta=eta_val)
            ax.semilogy(alpha_fine, th_curve, linestyle='--', color=colors[idx], alpha=0.5, linewidth=1.8)

        # Plot Simulated Dolinar points
        if not dolinar_df.empty:
            ax.semilogy(dolinar_df["alpha"], dolinar_df["ber"], marker='^', linestyle='-',
                        color=colors[idx], markersize=8.0, linewidth=1.6,
                        label=f'{label_name} (Dolinar)')

        # Plot Simulated Kennedy points (if requested)
        if show_kennedy and not kennedy_df.empty:
            ax.semilogy(kennedy_df["alpha"], kennedy_df["ber"], marker='x', linestyle=':',
                        color=colors[idx], markersize=7.5, linewidth=1.4, markeredgewidth=1.8,
                        label=f'{label_name} (Opt-Kennedy)')

    # Matched Typography & Sizing from plot_ideal_benchmark
    ax.set_title(title, fontsize=18, fontweight='bold', pad=14)
    ax.set_xlabel(r'Coherent State Amplitude $\alpha$', fontsize=16, labelpad=10)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, labelpad=10)
    
    # Exact Matched Ticks
    ax.tick_params(axis='both', which='major', labelsize=13.5, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3.5)
    
    ax.set_ylim(1e-7, 0.8)
    ax.set_xlim(0.1, 1.8)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Legend formatting
    legend_fontsize = 12 if n_sc > 6 else 14
    ax.legend(loc='lower left', fontsize=legend_fontsize, framealpha=0.92, edgecolor='gray')
    plt.tight_layout()

    # Save PNG and PDF
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    pdf_out = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_out, bbox_inches='tight')
    print(f"Saved Sweep Plot to:\n  • {output_file}\n  • {pdf_out}")
    plt.show()

# ==============================================================================
# CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a parameter sweep across any subset of scenarios.")
    parser.add_argument("data_source", nargs="?", default="combined_results.csv", help="Path to combined_results.csv or runs/ directory")
    parser.add_argument("-f", "--filter", default=None, help="Filter scenarios by keyword (e.g. 'Phase_Jitter', 'Latency', 'Efficiency', 'Dark_Counts', 'Dead_Time')")
    parser.add_argument("-i", "--ids", nargs="+", type=int, default=None, help="Filter by specific scenario IDs (e.g. -i 1 2 3 4 5 6)")
    parser.add_argument("-k", "--kennedy", action="store_true", help="Also plot Optimized Kennedy receiver curves")
    parser.add_argument("-t", "--title", default=None, help="Custom title for the plot")
    parser.add_argument("-o", "--output", default=None, help="Output image filename (.png)")

    args = parser.parse_args()
    plot_scenario_sweep(
        data_source=args.data_source,
        filter_keyword=args.filter,
        scenario_ids=args.ids,
        show_kennedy=args.kennedy,
        title=args.title,
        output_file=args.output
    )
