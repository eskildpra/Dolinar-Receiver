import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
import scipy.special as sp

def helstrom_bound(alpha, eta=1.0, pi0=0.5):
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

def simulate_dolinar_chunk(alpha, n_trials, eta, nu, pi0, latency_ratio, dead_time_ratio, phase_jitter_rad, steps=1000, T=1.0):
    pi1 = 1.0 - pi0
    dt = T / steps
    N_bar = 4.0 * alpha**2
    delay_steps = int(round(latency_ratio * steps))
    dead_steps = int(round(dead_time_ratio * steps))
    
    true_states = np.random.choice([0, 1], p=[pi0, pi1], size=n_trials).astype(np.int8)
    if phase_jitter_rad > 0:
        dphi = np.random.normal(0.0, phase_jitter_rad, size=n_trials).astype(np.float32)
    else:
        dphi = np.zeros(n_trials, dtype=np.float32)
        
    true_signals = np.where(true_states == 0, 0.0, np.sqrt(N_bar)).astype(np.float32) * np.exp(1j * dphi)
    clicks_real = np.zeros(n_trials, dtype=np.int16)
    dead_until = np.zeros(n_trials, dtype=np.int16)
    
    t_arr = np.linspace(dt, T, steps, dtype=np.float32)
    n_arr = N_bar * (t_arr / T)
    sqrt_term = np.sqrt(np.maximum(1.0 - 4.0 * pi0 * pi1 * np.exp(-eta * n_arr), 1e-10))
    J_arr = 0.5 * (1.0 - sqrt_term)
    u1_arr = -np.sqrt(N_bar) * (1.0 + J_arr / sqrt_term)
    u0_arr = np.sqrt(N_bar) * (J_arr / sqrt_term)
    
    # Memory-Safe Circular Ring Buffer (Takes only ~20 MB RAM)
    buf_len = delay_steps + 1
    ring_buf = np.zeros((buf_len, n_trials), dtype=np.int16)
    
    for k in range(steps):
        if delay_steps > 0:
            read_idx = k % buf_len
            applied_parity = ring_buf[read_idx]
        else:
            applied_parity = clicks_real
            
        u = np.where(applied_parity % 2 == 0, u1_arr[k], u0_arr[k])
        rate = eta * (np.abs(true_signals + u)**2) + nu
        p_click = 1.0 - np.exp(-rate * dt)
        
        can_click = (k >= dead_until)
        clicked = can_click & (np.random.rand(n_trials) < p_click)
        clicks_real += clicked.astype(np.int16)
        
        if delay_steps > 0:
            write_idx = (k + delay_steps) % buf_len
            ring_buf[write_idx] = clicks_real
            
        if dead_steps > 0:
            dead_until = np.where(clicked, k + dead_steps, dead_until)
            
    decisions = np.where(clicks_real % 2 == 0, 1, 0)
    return int(np.sum(decisions != true_states))

def simulate_kennedy_chunk(alpha, n_trials, eta, nu, pi0, dead_time_ratio, phase_jitter_rad, steps=1000, T=1.0):
    pi1 = 1.0 - pi0
    dt = T / steps
    dead_steps = int(round(dead_time_ratio * steps))
    
    true_states = np.random.choice([0, 1], p=[pi0, pi1], size=n_trials)
    dphi = np.random.normal(0.0, phase_jitter_rad, size=n_trials) if phase_jitter_rad > 0 else np.zeros(n_trials)
    true_alphas = np.where(true_states == 0, alpha, -alpha) * np.exp(1j * dphi)
    
    b_opt = get_optimal_kennedy_displacement(alpha, eta=eta, nu=nu, pi0=pi0, T=T)
    beta = -b_opt
    
    if dead_steps == 0:
        rate_true = (eta * np.abs(true_alphas + beta)**2 + nu) * T
        clicks = np.random.poisson(rate_true)
    else:
        clicks = np.zeros(n_trials, dtype=int)
        dead_until = np.zeros(n_trials, dtype=int)
        rate_step = eta * (np.abs(true_alphas + beta)**2) + nu
        p_click = 1.0 - np.exp(-rate_step * dt)
        for k in range(steps):
            can_click = (k >= dead_until)
            clicked = can_click & (np.random.rand(n_trials) < p_click)
            clicks += clicked.astype(int)
            dead_until = np.where(clicked, k + dead_steps, dead_until)
            
    decisions = np.where(clicks == 0, 0, 1)
    return int(np.sum(decisions != true_states))

if __name__ == "__main__":
    # Get Job Index from LSF (1 to 12) or from command-line argument
    job_idx = int(os.environ.get("LSB_JOBINDEX", sys.argv[1] if len(sys.argv) > 1 else 1))
    
    # Load Scenario Configuration
    with open("scenarios.json", "r") as f:
        scenarios = json.load(f)
        
    sc = scenarios[str(job_idx)]
    run_dir = os.path.join("runs", f"{job_idx:02d}_{sc['name']}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save exact run config into its folder
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(sc, f, indent=2)
        
    print(f"=== Starting Sub-Job {job_idx}: {sc['name']} on 16 Cores ===")
    print(f"Parameters: eta={sc['eta']}, nu={sc['nu']}, latency={sc['lat']}, dead_time={sc['dead']}, jitter={sc['jit_deg']} deg")
    
    alpha_grid = np.linspace(0.1, 1.8, 25)
    trials_per_point = 100_000_000           # 100 Million trials per alpha point
    n_jobs = 4                               # 4 CPU cores per task
    chunk_size = 2_500_000                   # 2.5 Million trials per task
    n_chunks = trials_per_point // chunk_size # 40 chunks total
    
    output_csv = os.path.join(run_dir, "results.csv")
    df_init = pd.DataFrame(columns=[
        "scenario", "receiver", "alpha", "eta", "nu", "latency", "dead_time", 
        "jitter_deg", "trials", "errors", "ber", "helstrom_bound"
    ])
    df_init.to_csv(output_csv, index=False)
    
    start_time = time.perf_counter()
    jit_rad = np.deg2rad(sc["jit_deg"])
    
    for alpha in alpha_grid:
        t0 = time.perf_counter()
        
        # 1. Dolinar Receiver
        d_errors = Parallel(n_jobs=n_jobs)(
            delayed(simulate_dolinar_chunk)(
                alpha, chunk_size, sc["eta"], sc["nu"], 0.5,
                sc["lat"], sc["dead"], jit_rad
            ) for _ in range(n_chunks)
        )
        total_d_err = sum(d_errors)
        d_ber = total_d_err / trials_per_point
        
        # 2. Optimized Kennedy Receiver
        k_errors = Parallel(n_jobs=n_jobs)(
            delayed(simulate_kennedy_chunk)(
                alpha, chunk_size, sc["eta"], sc["nu"], 0.5,
                sc["dead"], jit_rad
            ) for _ in range(n_chunks)
        )
        total_k_err = sum(k_errors)
        k_ber = total_k_err / trials_per_point
        
        h_bound = helstrom_bound(alpha, eta=sc["eta"])
        elapsed = time.perf_counter() - t0
        
        # Append record immediately (safe against crashes)
        records = [
            {"scenario": sc["name"], "receiver": "Dolinar", "alpha": alpha, "eta": sc["eta"], "nu": sc["nu"],
             "latency": sc["lat"], "dead_time": sc["dead"], "jitter_deg": sc["jit_deg"],
             "trials": trials_per_point, "errors": total_d_err, "ber": d_ber, "helstrom_bound": h_bound},
            {"scenario": sc["name"], "receiver": "Opt_Kennedy", "alpha": alpha, "eta": sc["eta"], "nu": sc["nu"],
             "latency": sc["lat"], "dead_time": sc["dead"], "jitter_deg": sc["jit_deg"],
             "trials": trials_per_point, "errors": total_k_err, "ber": k_ber, "helstrom_bound": h_bound}
        ]
        pd.DataFrame(records).to_csv(output_csv, mode='a', header=False, index=False)
        print(f"alpha={alpha:4.2f} | Dolinar={d_ber:.3e} | Kennedy={k_ber:.3e} | Time={elapsed:4.1f}s")
        
    print(f"=== Sub-Job {job_idx} finished in {(time.perf_counter() - start_time)/3600:.2f} hours ===")