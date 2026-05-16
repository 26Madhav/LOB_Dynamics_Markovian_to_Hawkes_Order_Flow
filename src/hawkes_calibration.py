import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

N_TYPES     = 4
N_SIZES     = 10
XI_BINS     = 6
MIN_EVENTS  = 10
N_RESTARTS  = 8
N_MAX       = 0.999
MU_FLOOR    = 1e-3
BETA_APPROX = 10.0

TYPE_LABELS = ["sell_submit", "buy_submit", "sell_cancel", "buy_cancel"]
XI_EDGES    = np.array([0, 1, 3, 6, 11, 21, np.inf])

def compute_excitement_bins(times: np.ndarray,
                             beta: float = BETA_APPROX) -> np.ndarray:
    n = len(times)
    if n == 0:
        return np.array([], dtype=int)
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + A[i - 1])
    return np.digitize(A, XI_EDGES[1:]).clip(0, XI_BINS - 1).astype(int)

def _negloglik(params, times, T):
    mu, n, beta = params
    alpha = n * beta
    if mu <= 0.0 or n <= 0.0 or beta <= 0.0:
        return 1e15
    k = len(times)
    if k == 0:
        return mu * T
    R = np.zeros(k)
    if k > 1:
        dt    = np.diff(times)
        decay = np.exp(-np.minimum(beta * dt, 500.0))
        for i in range(1, k):
            R[i] = decay[i - 1] * (1.0 + R[i - 1])
    tail        = T - times
    compensator = alpha * np.sum(1.0 - np.exp(-np.minimum(beta * tail, 500.0)))
    lam = mu + alpha * beta * R
    if np.any(lam <= 0.0):
        return 1e15
    return mu * T + compensator - np.sum(np.log(lam))

def fit_stream(times, T, rng):
    n_ev = len(times)
    if n_ev < MIN_EVENTS:
        mu_hat = max(n_ev / T, MU_FLOOR) if T > 0 else MU_FLOOR
        return dict(mu=mu_hat, alpha=0.0, beta=1.0, branching_ratio=0.0,
                    log_lik=float(-mu_hat*T + n_ev*np.log(mu_hat+1e-12)),
                    converged=False, fallback=True, n_events=n_ev)

    rate_guess = n_ev / T
    best_nll, best_x, best_ok = np.inf, None, False
    bounds = [(1e-8, None), (1e-8, N_MAX), (1e-3, None)]

    for _ in range(N_RESTARTS):
        x0 = [rng.uniform(0.2, 2.0) * rate_guess,
               rng.uniform(0.05, 0.75),
               np.log(2) / rng.uniform(0.2, 60.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(_negloglik, x0, args=(times, T), method="L-BFGS-B",
                           bounds=bounds,
                           options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9))
        if res.fun < best_nll:
            best_nll, best_x, best_ok = res.fun, res.x.copy(), res.success

    if best_x is None:
        mu_hat = max(rate_guess, MU_FLOOR)
        return dict(mu=mu_hat, alpha=0.0, beta=1.0, branching_ratio=0.0,
                    log_lik=np.nan, converged=False, fallback=True, n_events=n_ev)

    mu_hat, n_hat, beta_hat = best_x
    mu_hat = max(float(mu_hat), MU_FLOOR)
    return dict(mu=float(mu_hat), alpha=float(n_hat * beta_hat),
                beta=float(beta_hat), branching_ratio=float(n_hat),
                log_lik=float(-best_nll), converged=bool(best_ok),
                fallback=False, n_events=n_ev)

def calibrate_regime(data: dict, regime: int,
                     rng: np.random.Generator) -> tuple:
    mu_arr    = np.full((N_TYPES, N_SIZES, XI_BINS), MU_FLOOR)
    alpha_arr = np.zeros((N_TYPES, N_SIZES, XI_BINS))
    beta_arr  = np.ones((N_TYPES, N_SIZES, XI_BINS))
    diagnostics = []

    total = N_TYPES * N_SIZES * XI_BINS
    done  = 0
    t0    = time.time()

    for ti in range(N_TYPES):
        for zi in range(N_SIZES):
            z = zi + 1

            all_times = []
            T_session = 23400.0

            for ki in range(6):
                arr = data.get(f"R{regime}_type{ti}_z{z}_k{ki}_times", None)
                if arr is not None and len(arr) > 0:
                    all_times.append(arr)

                T_arr = data.get(f"R{regime}_type{ti}_z{z}_k{ki}_T", None)
                if T_arr is not None and len(T_arr) > 0:
                    T_session = float(T_arr[0])

            if not all_times:
                for xi_bin in range(XI_BINS):
                    done += 1
                    diagnostics.append(_diag_row(
                        regime, ti, zi, xi_bin,
                        MU_FLOOR, 0.0, 1.0, 0.0, 0, np.nan, False, True))
                continue

            times_all = np.sort(np.concatenate(all_times))

            times_all = times_all - times_all[0]
            times_all = np.clip(times_all, 0.0, T_session)

            xi_labels = compute_excitement_bins(times_all, beta=BETA_APPROX)

            for xi_bin in range(XI_BINS):
                times_xi = times_all[xi_labels == xi_bin]
                result   = fit_stream(times_xi, T_session, rng)

                mu_arr   [ti, zi, xi_bin] = result["mu"]
                alpha_arr[ti, zi, xi_bin] = result["alpha"]
                beta_arr [ti, zi, xi_bin] = result["beta"]

                diagnostics.append(_diag_row(
                    regime, ti, zi, xi_bin,
                    result["mu"], result["alpha"], result["beta"],
                    result["branching_ratio"], result["n_events"],
                    result["log_lik"], result["converged"], result["fallback"]))

                done += 1
                if done % 48 == 0:
                    elapsed = time.time() - t0
                    eta     = elapsed / done * (total - done)
                    print(f"    {100*done/total:5.1f}%  {done}/{total}  "
                          f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  |  "
                          f"{TYPE_LABELS[ti]} z={z} xi={xi_bin}  "
                          f"n_ev={result['n_events']}  "
                          f"mu={result['mu']:.4f}  "
                          f"n={result['branching_ratio']:.4f}"
                          + ("  [fallback]" if result["fallback"] else ""))

    print(f"  Done R{regime} in {time.time()-t0:.1f}s")
    return mu_arr, alpha_arr, beta_arr, diagnostics

def _diag_row(regime, ti, zi, xi_bin, mu, alpha, beta, n_ratio,
              n_ev, log_lik, converged, fallback):
    return dict(
        regime=f"R{regime}", type_idx=ti, type_label=TYPE_LABELS[ti],
        depth=zi+1, xi_bin=xi_bin, mu=mu, alpha=alpha, beta=beta,
        branching_ratio=n_ratio, n_events=n_ev, log_lik=log_lik,
        converged=converged, fallback=fallback)

def print_summary(diag_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    for regime in ["R1", "R2"]:
        sub    = diag_df[diag_df["regime"] == regime]
        hawkes = sub[~sub["fallback"]]
        print(f"\n  {regime}  ({len(sub)} streams total)")
        print(f"    Hawkes fitted   : {len(hawkes)} ({100*len(hawkes)/len(sub):.1f}%)")
        print(f"    Poisson fallback: {sub['fallback'].sum()}")
        print(f"    Converged       : {hawkes['converged'].sum()} / {len(hawkes)}")
        if len(hawkes) > 0:
            n = hawkes["branching_ratio"]
            print(f"    Branching ratio: mean={n.mean():.4f}  "
                  f"median={n.median():.4f}  max={n.max():.4f}")
            print(f"    n>0.5: {(n>0.5).sum()}  "
                  f"n>0.8: {(n>0.8).sum()} (near-critical)")
        mu_by_xi = sub.groupby("xi_bin")["mu"].mean()
        print(f"    Mean mu by xi_bin: " +
              "  ".join(f"xi={i}:{mu_by_xi.get(i, 0):.4f}"
                        for i in range(XI_BINS)))
        floor_count = (sub["mu"] <= MU_FLOOR + 1e-9).sum()
        print(f"    At mu floor: {floor_count} / {len(sub)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",  default="data/processed/interarrival_times.npz")
    parser.add_argument("--out",  default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("HAWKES LOB — hawkes_calibration.py  (v3, xi-binned)")
    print(f"  MU_FLOOR={MU_FLOOR}  XI_BINS={XI_BINS}  BETA_APPROX={BETA_APPROX}")
    print(f"  XI_EDGES: {XI_EDGES}")
    print("=" * 60)

    data = dict(np.load(args.npz, allow_pickle=False))
    print(f"Loaded {len(data)} arrays  ({len(data)//3} streams × 3 keys each)")

    all_diag, params = [], {}
    for regime in [1, 2]:
        print(f"\nCalibrating R{regime}...")
        mu, alpha, beta, diag = calibrate_regime(data, regime, rng)
        all_diag.extend(diag)
        params[f"R{regime}_mu"]    = mu
        params[f"R{regime}_alpha"] = alpha
        params[f"R{regime}_beta"]  = beta
        zero_mu = (mu <= 0).sum()
        if zero_mu:
            print(f"  WARNING: {zero_mu} mu values at zero after floor")

    np.savez(os.path.join(args.out, "hawkes_params.npz"), **params)
    print("\nSaved hawkes_params.npz")

    diag_df = pd.DataFrame(all_diag)
    diag_df.to_csv(os.path.join(args.out, "hawkes_diagnostics.csv"), index=False)
    print("Saved hawkes_diagnostics.csv")

    print_summary(diag_df)
    print("=" * 60)
    print("Complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
