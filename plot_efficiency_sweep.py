import sys
from plot_scenario_sweep import plot_scenario_sweep

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "combined_results.csv"
    kw = sys.argv[2] if len(sys.argv) > 2 else "Efficiency"
    plot_scenario_sweep(data_source=src, filter_keyword=kw)
