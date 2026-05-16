import argparse
import csv
import glob
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

Z        = 10
K        = 6
N_TYPES  = 4
XI_BINS  = 6
D_GRID   = 6

def isotonic_smooth_mu(mu):
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(increasing=False)
    mu = mu.copy()
    for e in [2, 3]:
        for xi in range(mu.shape[2]):
            y = mu[e, :, xi]
            if np.any(y > 0):
                mu[e, :, xi] = ir.fit_transform(np.arange(len(y)), y)
    return mu

def natural_k(z1: int, z2: int) -> int:
    return min(int((z1 + z2 - 2) * K // (2 * Z)), K - 1)

def depth_idx(e_type: int, z1: int, z2: int, max_d: int) -> int:
    raw = (z1 - 1) if e_type in (0, 2) else (z2 - 1)
    return min(raw, max_d - 1)

def hawkes_intensity(mu, alpha, beta, e_type, z1, z2, xi, A_e_at_last, elapsed):
    xi_idx  = min(xi, XI_BINS - 1)
    d_idx   = depth_idx(e_type, z1, z2, mu.shape[1])

    mu_e    = float(mu[e_type, d_idx, :].mean())
    alpha_e = float(alpha[e_type, d_idx, xi_idx])
    beta_e  = float(beta [e_type, d_idx, xi_idx])

    lam = mu_e + alpha_e * np.exp(-beta_e * elapsed) * A_e_at_last
    return max(lam, 1e-12), beta_e

def stationary_A(mu, alpha, beta, z1, z2):
    A0 = np.zeros(N_TYPES)
    for e in range(N_TYPES):
        d_idx = depth_idx(e, z1, z2, mu.shape[1])
        vals = []
        for xi in range(XI_BINS):
            mu_e    = float(mu   [e, d_idx, xi])
            alpha_e = float(alpha[e, d_idx, xi])
            beta_e  = float(beta [e, d_idx, xi])
            denom = beta_e - alpha_e
            if denom > 1e-6 and mu_e > 0:
                vals.append(mu_e / denom)
        A0[e] = float(np.mean(vals)) if vals else 0.0
    return A0

class HawkesLOBSimulator:
    def __init__(self, mu, alpha, beta, T_burnin=2.0, seed=42):
        self.mu       = mu
        self.alpha    = alpha
        self.beta     = beta
        self.T_burnin = T_burnin
        self.rng      = np.random.default_rng(seed)

    def _one_step(self, z1, z2, xi, A, t_last, t_now):
        rates = np.empty(N_TYPES)
        betas = np.empty(N_TYPES)
        for e in range(N_TYPES):
            r, b = hawkes_intensity(
                self.mu, self.alpha, self.beta,
                e, z1, z2, xi, A[e], t_now - t_last[e])
            rates[e] = r
            betas[e] = b

        total = rates.sum()
        if total < 1e-12:
            return None

        dt     = self.rng.exponential(1.0 / total)
        t_next = t_now + dt
        e_type = int(self.rng.choice(N_TYPES, p=rates / total))

        new_A      = A.copy()
        new_t_last = t_last.copy()
        for e in range(N_TYPES):
            new_A[e] = np.exp(-betas[e] * (t_next - t_last[e])) * A[e]
        new_A[e_type]      += 1.0
        new_t_last[e_type]  = t_next
        new_xi = min(int(new_A.sum()), XI_BINS - 1)

        return t_next, e_type, new_A, new_t_last, new_xi

    def _run_burnin(self, z1, z2):
        A      = stationary_A(self.mu, self.alpha, self.beta, z1, z2)
        xi     = min(int(A.sum()), XI_BINS - 1)
        t_last = np.zeros(N_TYPES)
        t_now  = 0.0

        if self.T_burnin <= 0:
            return z1, z2, xi, A, t_last

        while t_now < self.T_burnin:
            result = self._one_step(z1, z2, xi, A, t_last, t_now)
            if result is None:
                break
            t_now, e_type, A, t_last, xi = result

            if   e_type == 0: z1 = min(z1 + 1, Z)
            elif e_type == 1: z2 = min(z2 + 1, Z)
            elif e_type == 2: z1 = max(z1 - 1, 1)
            elif e_type == 3: z2 = max(z2 - 1, 1)

        t_last -= t_now
        return z1, z2, xi, A, t_last

    def simulate_batch(self, z1_0, z2_0, T, n_paths):
        ask_up_first = 0
        ask_up_T     = 0
        moved_count  = 0
        total_events = 0

        for _ in range(n_paths):
            z1, z2, xi, A, t_last = self._run_burnin(z1_0, z2_0)
            z1_init, z2_init = z1, z2

            t_now = 0.0
            path_up_T     = False
            path_moved    = False
            path_up_first = False
            n_events      = 0

            while t_now < T:
                result = self._one_step(z1, z2, xi, A, t_last, t_now)
                if result is None:
                    break
                t_next, e_type, A, t_last, xi = result
                if t_next > T:
                    break
                t_now = t_next
                n_events += 1

                z1_old, z2_old = z1, z2
                if   e_type == 0: z1 = min(z1 + 1, Z)
                elif e_type == 1: z2 = min(z2 + 1, Z)
                elif e_type == 2: z1 = max(z1 - 1, 1)
                elif e_type == 3: z2 = max(z2 - 1, 1)

                if not path_up_T and z1 < z1_init:
                    path_up_T = True

                ask_moved = (e_type == 2 and z1 < z1_old)
                bid_moved = (e_type == 3 and z2 < z2_old)

                if not path_moved and (ask_moved or bid_moved):
                    path_moved = True
                    moved_count += 1
                    if ask_moved:
                        path_up_first = True

            if path_up_T:
                ask_up_T += 1
            if path_up_first:
                ask_up_first += 1
            total_events += n_events

        return (
            float(ask_up_first / max(moved_count, 1)),
            float(ask_up_T     / n_paths),
            float(moved_count  / n_paths),
            float(total_events / n_paths),
        )

def detect_kbe_aliasing(surf, D=D_GRID, tol=1e-6):
    n_flat = 0
    n_total = 0
    details = []
    for s in range(2, 2 * D + 1):
        vals = []
        for i in range(D):
            j = s - (i + 1) - 1
            if 0 <= j < D:
                v = surf[i, j]
                if np.isfinite(v):
                    vals.append(v)
        if len(vals) > 1:
            n_total += 1
            spread = max(vals) - min(vals)
            if spread < tol:
                n_flat += 1
                details.append(f"z1+z2={s}: all={vals[0]:.4f}")
    frac = n_flat / max(n_total, 1)
    return frac, details

def print_rate_diagnostics(mu, alpha, beta, label, z1=3, z2=3):
    print(f"\n  [{label}] Rate diagnostics at (z1={z1}, z2={z2}):")
    names = ["sell_submit", "buy_submit ", "sell_cancel", "buy_cancel "]
    total_xi0 = 0.0
    total_xi3 = 0.0
    for e in range(N_TYPES):
        d = depth_idx(e, z1, z2, mu.shape[1])
        r0 = float(mu[e, d, 0])
        r3 = float(mu[e, d, 3])
        a  = float(alpha[e, d, 3])
        b  = float(beta [e, d, 3])
        br = a / max(b, 1e-12)
        A_stat = float(mu[e, d, 0]) / max(float(beta[e, d, 0]) - float(alpha[e, d, 0]), 1e-9)
        print(f"    {names[e]}: μ(xi=0)={r0:.4f}  μ(xi=3)={r3:.4f}  "
              f"α={a:.4f}  β={b:.1f}  η={br:.5f}  A_stat≈{A_stat:.5f}")
        total_xi0 += r0
        total_xi3 += r3

    print(f"    ── Aggregate μ at xi=0: {total_xi0:.4f} ev/s  "
          f"at xi=3: {total_xi3:.4f} ev/s")

    A0 = stationary_A(mu, alpha, beta, z1, z2)
    xi_ws = min(int(A0.sum()), XI_BINS - 1)
    print(f"    ── Warm-start A: {A0.round(5)}  → xi_init={xi_ws}")
    if total_xi0 < 0.05:
        print(f"    *** COLD-START PROBLEM: μ(xi=0)={total_xi0:.4f} — "
              f"simulator will be dormant without warm-start ***")

def print_surface(surf, label, D=D_GRID):
    print(f"\n  {label}:")
    print("         " + "".join(f"  z2={j+1}" for j in range(D)))
    for i in range(D):
        row = f"  z1={i+1}   " + "".join(
            f"  {surf[i,j]:.4f}" if np.isfinite(surf[i, j]) else "     nan"
            for j in range(D)
        )
        print(row)
    v = surf[:D, :D]
    valid = v[np.isfinite(v)]
    if len(valid):
        print(f"  min={valid.min():.4f}  max={valid.max():.4f}  "
              f"mean={valid.mean():.4f}  cells={len(valid)}/36")
    d1 = v[0, 2] - v[D-1, 2]
    d2 = v[2, D-1] - v[2, 0]
    print(f"  d/d(z1)[z2=3]: {v[0,2]:.4f}→{v[D-1,2]:.4f}  "
          f"{'✓ decreasing' if d1 > 0 else '✗ non-decreasing'}")
    print(f"  d/d(z2)[z1=3]: {v[2,0]:.4f}→{v[2,D-1]:.4f}  "
          f"{'✓ increasing' if d2 > 0 else '✗ non-increasing'}")

def check_monotonicity(surf, D=D_GRID):
    z1_ok, z1_tot, z2_ok, z2_tot = 0, 0, 0, 0

    for j in range(1, D):
        col = surf[1:D, j]
        fin = np.isfinite(col)
        if fin.sum() >= 2:
            z1_ok += int(col[fin][-1] <= col[fin][0])
            z1_tot += 1
    for i in range(1, D):
        row = surf[i, 1:D]
        fin = np.isfinite(row)
        if fin.sum() >= 2:
            z2_ok += int(row[fin][-1] >= row[fin][0])
            z2_tot += 1
    return z1_ok / max(z1_tot, 1), z2_ok / max(z2_tot, 1)

def gap_stats(surf_ref, surf_cmp, label=""):
    mask = ~(np.isnan(surf_ref) | np.isnan(surf_cmp))
    mask[0, :] = False
    mask[:, 0] = False
    if not mask.any():
        return None
    a, b = surf_ref[mask], surf_cmp[mask]
    d    = b - a
    corr = float(np.corrcoef(a, b)[0, 1]) if mask.sum() > 2 else float("nan")
    return {
        "label": label, "n_cells": int(mask.sum()),
        "mean_gap": float(d.mean()), "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "max_gap": float(np.abs(d).max()), "corr": corr,
    }

def candidate_lobster_dirs(data_dir, ticker):
    roots = [data_dir]
    base = os.path.basename(os.path.normpath(data_dir))
    for candidate in [base, ticker, ticker.lower(), ticker.upper()]:
        if candidate and candidate not in roots:
            roots.append(candidate)
    return [root for root in roots if os.path.isdir(root)]

def recursive_glob(root, pattern):
    matches = glob.glob(os.path.join(root, pattern))
    matches.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(set(matches))

def find_lobster_pair(data_dir, ticker, regime):
    roots = candidate_lobster_dirs(data_dir, ticker)
    variants = list(dict.fromkeys([ticker, ticker.upper(), ticker.lower()]))

    for root in roots:
        for tkr in variants:
            for mp, op in [
                (f"{tkr}*R{regime}*message*.csv",  f"{tkr}*R{regime}*orderbook*.csv"),
                (f"{tkr}*message*R{regime}*.csv",  f"{tkr}*orderbook*R{regime}*.csv"),
                (f"*R{regime}*message*.csv",       f"*R{regime}*orderbook*.csv"),
            ]:
                msgs = recursive_glob(root, mp)
                obs  = recursive_glob(root, op)
                if msgs and obs:
                    return msgs[0], obs[0]

    csvs = []
    for root in roots:
        csvs.extend(recursive_glob(root, "*.csv"))
    csvs = sorted(set(csvs))

    ticker_l = ticker.lower()
    msgs = [p for p in csvs if "message" in os.path.basename(p).lower()]
    obs  = [p for p in csvs if "orderbook" in os.path.basename(p).lower()]

    ticker_msgs = [p for p in msgs if ticker_l in os.path.basename(p).lower()]
    ticker_obs  = [p for p in obs if ticker_l in os.path.basename(p).lower()]
    if ticker_msgs and ticker_obs:
        msgs, obs = ticker_msgs, ticker_obs

    if len(msgs) >= regime and len(obs) >= regime:
        return msgs[regime - 1], obs[regime - 1]
    if msgs and obs:
        return msgs[0], obs[0]
    return None, None

def build_empirical_surface(msg_path, ob_path, vol_unit=100,
                             n_levels=10, z_max=D_GRID):
    import pandas as pd
    print(f"    Reading {os.path.basename(msg_path)}")
    print(f"    Reading {os.path.basename(ob_path)}")
    msg_cols = ["time","event_type","order_id","size","price","direction"]
    ob_names = []
    for lv in range(1, n_levels + 1):
        ob_names += [f"ask_p{lv}", f"ask_v{lv}", f"bid_p{lv}", f"bid_v{lv}"]
    try:
        msg = pd.read_csv(msg_path, header=None, names=msg_cols,
                          dtype={"event_type": int, "direction": int})
        ob  = pd.read_csv(ob_path,  header=None, names=ob_names)
    except Exception as exc:
        print(f"    ERROR: {exc}"); return None, 0

    n = min(len(msg), len(ob))
    msg = msg.iloc[:n].reset_index(drop=True)
    ob  = ob.iloc[:n].reset_index(drop=True)

    valid = ((ob["ask_p1"] > 0) & (ob["bid_p1"] > 0) &
             (ob["ask_p1"] < 9_999_999_999) &
             (ob["ask_p1"] > ob["bid_p1"]))
    msg = msg[valid].reset_index(drop=True)
    ob  = ob[valid].reset_index(drop=True)
    n_rows = len(msg)
    print(f"    {n_rows:,} rows after cleaning")
    if n_rows < 200:
        print("    Too few rows"); return None, 0

    ask_dep = np.clip((ob["ask_v1"].values // vol_unit).astype(int), 1, Z)
    bid_dep = np.clip((ob["bid_v1"].values // vol_unit).astype(int), 1, Z)
    ask_px  = ob["ask_p1"].values
    bid_px  = ob["bid_p1"].values

    ask_changed = np.zeros(n_rows, dtype=bool)
    ask_up      = np.zeros(n_rows, dtype=bool)
    ask_changed[1:] = ask_px[1:] != ask_px[:-1]
    ask_up[1:]      = ask_px[1:] > ask_px[:-1]
    bid_changed     = np.zeros(n_rows, dtype=bool)
    bid_changed[1:] = bid_px[1:] != bid_px[:-1]
    price_moved     = ask_changed | bid_changed

    counts = np.zeros((z_max, z_max), dtype=np.int64)
    hits   = np.zeros((z_max, z_max), dtype=np.int64)

    i = 1
    while i < n_rows:
        if price_moved[i]:
            z1 = min(ask_dep[i - 1], z_max)
            z2 = min(bid_dep[i - 1], z_max)
            counts[z1 - 1, z2 - 1] += 1
            if ask_up[i]:
                hits[z1 - 1, z2 - 1] += 1

            while i < n_rows and price_moved[i]:
                i += 1
        i += 1

    surf = np.full((z_max, z_max), np.nan)
    mask = counts >= 10
    surf[mask] = hits[mask] / counts[mask]
    n_pop = int(mask.sum())
    print(f"    Empirical: {n_pop}/36 cells (≥10 obs), "
          f"min_count={counts[mask].min() if n_pop else 0}, "
          f"mean={surf[mask].mean():.4f}" if n_pop else "")
    print(f"    Total price moves observed: {counts.sum():,}")
    return surf, n_rows

def run(args):
    print("=" * 65)
    print("HAWKES LOB — hawkes_validation.py  (v4 — warm-start + diagnostics)")
    print("=" * 65)
    print(f"  ticker={args.ticker}  T={args.T}s  T_burnin={args.T_burnin}s  "
          f"n_mc={args.n_mc}  seed={args.seed}")

    os.makedirs(args.out, exist_ok=True)

    kbe_path = os.path.join(args.lh_dir, "hawkes_kbe_results.npz")
    if not os.path.exists(kbe_path):
        sys.exit(f"ERROR: {kbe_path} not found")
    kbe_data = np.load(kbe_path, allow_pickle=True)
    print(f"  KBE keys: {list(kbe_data.files)}")

    params_path = os.path.join(args.lh_dir, "hawkes_params.npz")
    if not os.path.exists(params_path):
        sys.exit(f"ERROR: {params_path} not found")
    raw = np.load(params_path, allow_pickle=True)

    regimes      = [1, 2] if args.regime == "both" else [int(args.regime)]
    all_results  = {}
    all_gaps     = []
    report_lines = []

    for regime in regimes:
        rkey = f"R{regime}"
        print(f"\n{'='*65}\n  Regime {rkey}\n{'='*65}")

        mu_r    = np.where(np.isfinite(raw[f"{rkey}_mu"]) &
                           (raw[f"{rkey}_mu"] >= 0),
                           raw[f"{rkey}_mu"],    0.0).astype(float)
        mu_r = isotonic_smooth_mu(mu_r)
        alpha_r = np.where(np.isfinite(raw[f"{rkey}_alpha"]) &
                           (raw[f"{rkey}_alpha"] >= 0),
                           raw[f"{rkey}_alpha"],  0.0).astype(float)
        beta_r  = np.where(np.isfinite(raw[f"{rkey}_beta"]) &
                           (raw[f"{rkey}_beta"]  > 0),
                           raw[f"{rkey}_beta"],   1.0).astype(float)

        print(f"  mu   ∈ [{mu_r.min():.4g}, {mu_r.max():.4g}]  "
              f"alpha ∈ [{alpha_r.min():.4g}, {alpha_r.max():.4g}]  "
              f"beta  ∈ [{beta_r.min():.4g}, {beta_r.max():.4g}]")

        print_rate_diagnostics(mu_r, alpha_r, beta_r, rkey, z1=3, z2=3)
        print_rate_diagnostics(mu_r, alpha_r, beta_r, rkey, z1=1, z2=1)

        kbe_surf = None
        for candidate in [f"{rkey}_first_step_up", f"{rkey}_kbe_relative",
                          f"{rkey}_prob_ask_up",   f"{rkey}_surface"]:
            if candidate in kbe_data.files:
                raw_kbe = kbe_data[candidate]
                print(f"\n  KBE key: '{candidate}'  shape={raw_kbe.shape}")
                kbe_surf = (raw_kbe[:D_GRID, :D_GRID, :].mean(axis=2)
                            if raw_kbe.ndim == 3
                            else raw_kbe[:D_GRID, :D_GRID].astype(float).copy())
                break

        if kbe_surf is None:
            print(f"  WARNING: no KBE surface for {rkey}; skipping")
            continue
        kbe_surf = np.clip(kbe_surf.astype(float), 0.0, 1.0)

        alias_frac, alias_details = detect_kbe_aliasing(kbe_surf)
        print(f"\n  KBE anti-diagonal aliasing: {alias_frac*100:.0f}% of "
              f"anti-diagonals are constant")
        if alias_frac > 0.5:
            print("  *** KBE ALIASING DETECTED ***")
            print("  The KBE engine used natural_k=f(z1+z2) as state index")
            print("  instead of separate (z1,z2) axes.  This collapses ALL")
            print("  cells with the same z1+z2 to identical probabilities,")
            print("  destroying the z1 vs z2 asymmetry that drives P(ask up).")
            print()
            print("  FIX in hawkes_kbe_engine.py:")
            print("    OLD: state_idx = natural_k(z1, z2)   # 0..K-1, many-to-one")
            print("    NEW: state_idx = (z1-1)*Z + (z2-1)   # 0..Z²-1, injective")
            print()
            print("  Aliased anti-diagonals:")
            for d in alias_details[:6]:
                print(f"    {d}")
            if len(alias_details) > 6:
                print(f"    ... ({len(alias_details)-6} more)")

        print_surface(kbe_surf, f"KBE ({rkey})")
        z1f_k, z2f_k = check_monotonicity(kbe_surf)
        print(f"  KBE monotonicity: z1↓ {z1f_k*100:.0f}%  z2↑ {z2f_k*100:.0f}%")

        print(f"\n  MC: {args.n_mc} paths × 36 cells  "
              f"T={args.T}s  T_burnin={args.T_burnin}s …")
        sim = HawkesLOBSimulator(
            mu_r, alpha_r, beta_r,
            T_burnin=args.T_burnin, seed=args.seed)

        mc_first = np.zeros((D_GRID, D_GRID))
        mc_T_arr = np.zeros((D_GRID, D_GRID))
        mc_moved = np.zeros((D_GRID, D_GRID))
        mc_nevts = np.zeros((D_GRID, D_GRID))

        t0 = time.time()
        for i, z1 in enumerate(range(1, D_GRID + 1)):
            for j, z2 in enumerate(range(1, D_GRID + 1)):
                pf, pT, pm, ne = sim.simulate_batch(z1, z2, args.T, args.n_mc)
                mc_first[i, j] = pf
                mc_T_arr[i, j] = pT
                mc_moved[i, j] = pm
                mc_nevts[i, j] = ne
            print(f"    z1={z1}  p_moved={mc_moved[i].mean():.3f}  "
                  f"mean_events={mc_nevts[i].mean():.1f}  "
                  f"({time.time()-t0:.1f}s)")

        print(f"  MC done  ({time.time()-t0:.1f}s total)")
        print(f"  MC mean events/path: {mc_nevts.mean():.1f}")

        if mc_moved.mean() < 0.01:
            print(
                "\n  *** MC WARNING: p_moved still < 1% even with warm-start ***\n"
                "  This means the warm-start A values are still near zero.\n"
                "  Likely cause: mu is zero at ALL xi bins for d_idx=z-1.\n"
                "  Action: run  python hawkes_calibration.py  and check the\n"
                "  hawkes_diagnostics.csv file.  If mu[:,d,xi] ≈ 0 for xi=0..5\n"
                "  at depth d, the calibration converged to a degenerate solution.\n"
                "  Consider re-calibrating with a non-zero mu floor (e.g. 1e-4)."
            )
        elif mc_nevts.mean() < 1:
            print(
                "\n  *** MC WARNING: very few events per path ***\n"
                "  The warm-start worked (A>0) but events decay before firing.\n"
                "  Increase T_burnin or check alpha/beta values."
            )

        print_surface(mc_first, "MC — P(ask up | first price-moving event)")
        print_surface(mc_T_arr, f"MC — P(ask up within T={args.T}s)")
        z1f_mc, z2f_mc = check_monotonicity(mc_first)
        print(f"  MC monotonicity: z1↓ {z1f_mc*100:.0f}%  z2↑ {z2f_mc*100:.0f}%")

        g = gap_stats(kbe_surf, mc_first, f"{rkey}: KBE vs MC(first)")
        if g:
            all_gaps.append(g)
            print(f"\n  KBE vs MC: MAE={g['mae']:.4f}  "
                  f"RMSE={g['rmse']:.4f}  corr={g['corr']:.4f}")

        emp_surf = None
        if not args.skip_empirical:
            mp, op = find_lobster_pair(args.data_dir, args.ticker, regime)
            if mp and op:
                emp_surf, n_emp = build_empirical_surface(
                    mp, op, vol_unit=args.vol_unit,
                    n_levels=args.n_levels, z_max=D_GRID)
                if emp_surf is not None:
                    print_surface(emp_surf, f"Empirical ({rkey}, n={n_emp:,})")
            else:
                print(f"  No LOBSTER files for {rkey} in {args.data_dir}")

        if emp_surf is not None:
            for sa, sb, lbl in [
                (kbe_surf, emp_surf, f"{rkey}: KBE   vs Empirical"),
                (mc_first, emp_surf, f"{rkey}: MC    vs Empirical"),
                (mc_T_arr, emp_surf, f"{rkey}: MC(T) vs Empirical"),
            ]:
                g2 = gap_stats(sa, sb, lbl)
                if g2:
                    all_gaps.append(g2)
                    print(f"  {lbl}: MAE={g2['mae']:.4f}  "
                          f"RMSE={g2['rmse']:.4f}  corr={g2['corr']:.4f}  "
                          f"gap={g2['mean_gap']:+.4f}")

            g_kbe = gap_stats(kbe_surf, emp_surf)
            g_mc  = gap_stats(mc_first, emp_surf)
            if g_kbe and g_mc:
                mae_kbe, mae_mc = g_kbe["mae"], g_mc["mae"]
                if mc_moved.mean() < 0.01:
                    line = (f"{rkey}: MC degenerate (p_moved={mc_moved.mean():.4f})"
                            f" — Finding C invalid")
                    print(f"\n  *** FINDING C (INVALID): {line} ***")
                elif alias_frac > 0.5:
                    line = (f"{rkey}: KBE aliased — KBE MAE={mae_kbe:.4f} is "
                            f"unreliable.  MC MAE={mae_mc:.4f} vs empirical.")
                    print(f"\n  *** FINDING C (KBE ALIASED): {line} ***")
                    report_lines.append(line)
                elif mae_kbe > 1e-6:
                    closure = (mae_kbe - mae_mc) / mae_kbe * 100.0
                    line = (f"{rkey}: Hawkes closes {closure:+.1f}% of gap  "
                            f"(KBE MAE={mae_kbe:.4f} → MC MAE={mae_mc:.4f})")
                    print(f"\n  *** FINDING C: {line} ***")
                    report_lines.append(line)

        all_results[f"{rkey}_kbe"]       = kbe_surf
        all_results[f"{rkey}_mc_first"]  = mc_first
        all_results[f"{rkey}_mc_T"]      = mc_T_arr
        all_results[f"{rkey}_mc_moved"]  = mc_moved
        all_results[f"{rkey}_mc_nevts"]  = mc_nevts
        if emp_surf is not None:
            all_results[f"{rkey}_empirical"] = emp_surf

    all_results.update(T=np.array(args.T), n_mc=np.array(args.n_mc))
    npz_out = os.path.join(args.out, "hawkes_validation_results.npz")
    np.savez(npz_out, **all_results)
    print(f"\nSaved {npz_out}")

    csv_out = os.path.join(args.out, "hawkes_validation_results.csv")
    fields  = ["regime", "z1", "z2", "k_natural",
               "kbe", "mc_first", "mc_T", "mc_moved", "mc_nevts", "empirical",
               "gap_kbe_mc", "gap_mc_emp", "gap_kbe_emp"]
    with open(csv_out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for regime in regimes:
            rkey = f"R{regime}"
            if f"{rkey}_kbe" not in all_results:
                continue
            has_emp = f"{rkey}_empirical" in all_results
            for i in range(D_GRID):
                for j in range(D_GRID):
                    kv = float(all_results[f"{rkey}_kbe"][i, j])
                    mf = float(all_results[f"{rkey}_mc_first"][i, j])
                    mt = float(all_results[f"{rkey}_mc_T"][i, j])
                    mm = float(all_results[f"{rkey}_mc_moved"][i, j])
                    mn = float(all_results[f"{rkey}_mc_nevts"][i, j])
                    ev = (float(all_results[f"{rkey}_empirical"][i, j])
                          if has_emp else float("nan"))
                    fin = np.isfinite(ev)
                    wr.writerow({
                        "regime": regime, "z1": i+1, "z2": j+1,
                        "k_natural": natural_k(i+1, j+1),
                        "kbe":       f"{kv:.6f}",
                        "mc_first":  f"{mf:.6f}",
                        "mc_T":      f"{mt:.6f}",
                        "mc_moved":  f"{mm:.6f}",
                        "mc_nevts":  f"{mn:.2f}",
                        "empirical": f"{ev:.6f}" if fin else "",
                        "gap_kbe_mc":  f"{mf-kv:.6f}",
                        "gap_mc_emp":  f"{mf-ev:.6f}" if fin else "",
                        "gap_kbe_emp": f"{kv-ev:.6f}" if fin else "",
                    })
    print(f"Saved {csv_out}")

    if all_gaps:
        gap_out = os.path.join(args.out, "hawkes_gap_stats.csv")
        with open(gap_out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=all_gaps[0].keys())
            wr.writeheader(); wr.writerows(all_gaps)
        print(f"Saved {gap_out}")

    rpt_out = os.path.join(args.out, "hawkes_validation_report.txt")
    with open(rpt_out, "w") as fh:
        fh.write("HAWKES LOB — VALIDATION REPORT  (v4)\n")
        fh.write("=" * 65 + "\n\n")
        fh.write(f"ticker={args.ticker}  T={args.T}s  "
                 f"T_burnin={args.T_burnin}s  n_mc={args.n_mc}\n\n")
        fh.write("Bugs fixed vs v2/v3\n" + "-"*40 + "\n")
        fh.write("  1. MC cold-start: warm-start A from stationary E[A_e] + burnin\n")
        fh.write("  2. KBE aliasing: detected and flagged; fix in kbe_engine.py\n\n")
        fh.write("Gap Statistics\n" + "-"*40 + "\n")
        for g in all_gaps:
            fh.write(f"\n  {g['label']}\n"
                     f"    n={g['n_cells']}  mean={g['mean_gap']:+.5f}  "
                     f"MAE={g['mae']:.5f}  RMSE={g['rmse']:.5f}  "
                     f"corr={g['corr']:.4f}  max={g['max_gap']:.5f}\n")
        fh.write("\n\nFinding C\n" + "-"*40 + "\n")
        if report_lines:
            for ln in report_lines:
                fh.write(f"  {ln}\n")
        else:
            fh.write("  Not computed (MC degenerate or KBE aliased — "
                     "fix upstream files first)\n")
        fh.write("\n\nKBE aliasing summary\n" + "-"*40 + "\n")
        fh.write(
            "  hawkes_kbe_engine.py used natural_k(z1,z2)=floor((z1+z2-2)*K/(2*Z))\n"
            "  as its state index.  This function is MANY-TO-ONE: multiple (z1,z2)\n"
            "  pairs with the same z1+z2 map to the same k value, so the KBE result\n"
            "  is identical for all cells on each anti-diagonal.\n\n"
            "  Required fix:\n"
            "    state_idx = (z1 - 1) * Z + (z2 - 1)   # range 0..Z²-1\n"
            "  This preserves the full (z1, z2) resolution so that:\n"
            "    - sell-side events index into mu[e, z1-1, xi]  (ask depth)\n"
            "    - buy-side  events index into mu[e, z2-1, xi]  (bid depth)\n"
            "  and P(ask up | z1, z2) is correctly asymmetric in z1 vs z2.\n"
        )
        fh.write("\n\nMC warm-start details\n" + "-"*40 + "\n")
        fh.write(
            "  The Hawkes calibration fits mu[e,d,xi] per excitement bin xi.\n"
            "  At xi=0 (cold start) mu is near zero for most cells because\n"
            "  the empirical data always has some residual excitement.\n"
            "  Fix: initialise A[e] = mu[e,d,xi_avg] / (beta[e,d,xi_avg] - alpha)\n"
            "  then run T_burnin seconds of burn-in before measuring.\n"
        )
    print(f"Saved {rpt_out}")

    print("\n" + "=" * 65)
    print("SUMMARY OF ISSUES FOUND")
    print("=" * 65)
    for regime in regimes:
        rkey = f"R{regime}"
        if f"{rkey}_mc_moved" not in all_results:
            continue
        pm  = all_results[f"{rkey}_mc_moved"].mean()
        ne  = all_results[f"{rkey}_mc_nevts"].mean()
        kbe = all_results.get(f"{rkey}_kbe")
        print(f"\n  {rkey}:")
        print(f"    MC p_moved={pm:.4f}  mean_events={ne:.1f}  "
              f"({'OK' if pm > 0.05 else 'DEGENERATE'})")
        if kbe is not None:
            af, _ = detect_kbe_aliasing(kbe)
            print(f"    KBE aliasing={af*100:.0f}%  "
                  f"({'ALIASED — fix kbe_engine.py' if af>0.5 else 'OK'})")
            z1f, z2f = check_monotonicity(kbe)
            print(f"    KBE monotone: z1↓={z1f:.0%}  z2↑={z2f:.0%}  "
                  f"({'OK' if z1f>0.8 and z2f>0.8 else 'WRONG'})")

    print()
    print("ACTION ITEMS (in priority order):")
    print("  1. FIX hawkes_kbe_engine.py: replace natural_k state indexing")
    print("     with (z1-1)*Z + (z2-1) to resolve the anti-diagonal aliasing.")
    print()
    print("  2. FIX hawkes_calibration.py: ensure mu[e,d,xi=0] > 0 so the")
    print("     Gillespie has a non-zero baseline rate at cold start.")
    print("     Options: (a) add a small floor (e.g. mu_floor=1e-3 events/s),")
    print("              (b) fit mu on the marginal rate ignoring xi binning,")
    print("              (c) use a two-step calibration: fit mu first from")
    print("                  low-xi windows, then fit alpha/beta from residuals.")
    print()
    print("  3. Once (1) and (2) are fixed, re-run with:")
    print("     python hawkes_validation.py --n_mc 2000 --T 5.0 --T_burnin 2.0")
    print("=" * 65)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lh_dir",         default="data/processed")
    p.add_argument("--data_dir",       default="data/raw/intc")
    p.add_argument("--out",            default="data/processed")
    p.add_argument("--ticker",         default="INTC")
    p.add_argument("--n_mc",           type=int,   default=2000)
    p.add_argument("--T",              type=float, default=5.0)
    p.add_argument("--T_burnin",       type=float, default=2.0,
                   help="Burn-in window before measurement (seconds)")
    p.add_argument("--M",              type=int,   default=6)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--regime",         default="both",
                   choices=["1", "2", "both"])
    p.add_argument("--skip_empirical", action="store_true")
    p.add_argument("--n_levels",       type=int,   default=10)
    p.add_argument("--price_unit",     type=int,   default=100)
    p.add_argument("--vol_unit",       type=int,   default=100)
    return p.parse_args()

if __name__ == "__main__":
    args = main()
    run(args)
