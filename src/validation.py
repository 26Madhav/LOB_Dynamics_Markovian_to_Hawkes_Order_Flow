import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import warnings

from data_loader import (
    load_lobster, LOBData,
    MAX_SIZE_UNITS, SHARE_UNIT,
)
from calibration import (
    CalibratedParams, calibrate_fast,
    RegimeRates, N_SIZES, N_LEVELS,
)
from kbe_engine import (
    ProbabilityTable, run_kbe,
    N_STATES, DT, T_HORIZON,
)

N_MC         = 500
MC_MAX_STEPS = 10_000
MC_MAX_TIME  = 5.0

@dataclass
class ValidationResults:
    kbe_R1     : np.ndarray
    kbe_R2     : np.ndarray

    mc_R1      : np.ndarray
    mc_R2      : np.ndarray
    mc_std_R1  : np.ndarray
    mc_std_R2  : np.ndarray
    mc_n_R1    : np.ndarray
    mc_n_R2    : np.ndarray

    emp_R1     : np.ndarray
    emp_R2     : np.ndarray
    emp_n_R1   : np.ndarray
    emp_n_R2   : np.ndarray

    def save(self, path: str = 'results/validation_results.npz') -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v for k, v in self.__dict__.items()})
        print(f"  Saved validation results → {path}")

    def to_dataframe(self, regime: int) -> pd.DataFrame:
        rows = []
        kbe = self.kbe_R1  if regime == 1 else self.kbe_R2
        mc  = self.mc_R1   if regime == 1 else self.mc_R2
        std = self.mc_std_R1 if regime == 1 else self.mc_std_R2
        emp = self.emp_R1  if regime == 1 else self.emp_R2
        n   = self.emp_n_R1 if regime == 1 else self.emp_n_R2

        for i in range(N_STATES):
            for j in range(N_STATES):
                rows.append({
                    'z1'        : i + 1,
                    'z2'        : j + 1,
                    'regime'    : regime,
                    'KBE'       : kbe[i, j],
                    'MC'        : mc[i, j],
                    'MC_std'    : std[i, j],
                    'Empirical' : emp[i, j],
                    'Emp_n'     : n[i, j],
                    'KBE_MC_err': abs(kbe[i,j] - mc[i,j]),
                    'KBE_Emp_err': abs(kbe[i,j] - emp[i,j]),
                })
        return pd.DataFrame(rows)

def run_monte_carlo(
    params  : CalibratedParams,
    n_mc    : int  = N_MC,
    verbose : bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    if verbose:
        print("=" * 60)
        print("MONTE CARLO SIMULATION")
        print("=" * 60)
        print(f"  Replications per state: {n_mc}")
        print(f"  Total states: {N_STATES}×{N_STATES}×2 = "
              f"{2*N_STATES*N_STATES} state-regime pairs")
        print(f"  Estimated total runs: "
              f"{2*N_STATES*N_STATES*n_mc:,}")

    results = {}
    for regime, r in [(1, params.R1), (2, params.R2)]:
        if verbose:
            print(f"\n  Simulating Regime R{regime}...")

        mc_surface  = np.zeros((N_STATES, N_STATES))
        std_surface = np.zeros((N_STATES, N_STATES))
        n_surface   = np.zeros((N_STATES, N_STATES), dtype=int)

        rate_struct = _build_rate_structure(r)

        state_pairs = [(z1, z2)
                       for z1 in range(1, N_STATES+1)
                       for z2 in range(1, N_STATES+1)]

        iterator = tqdm(state_pairs, desc=f"  R{regime}",
                        disable=not verbose)

        for z1, z2 in iterator:
            outcomes = _simulate_single_state(
                z1, z2, r, rate_struct, n_mc
            )
            i, j = z1 - 1, z2 - 1
            mc_surface[i, j]  = np.mean(outcomes)
            std_surface[i, j] = np.std(outcomes) / np.sqrt(len(outcomes))
            n_surface[i, j]   = len(outcomes)

        results[regime] = (mc_surface, std_surface, n_surface)

        if verbose:
            print(f"  R{regime} MC surface:")
            _print_surface(mc_surface, label=f"MC R{regime}",
                           std=std_surface)

    mc_R1,  std_R1, n_R1 = results[1]
    mc_R2,  std_R2, n_R2 = results[2]
    return mc_R1, mc_R2, std_R1, std_R2, n_R1, n_R2

def _build_rate_structure(r: RegimeRates) -> dict:
    mkt_fraction = 0.062 if r.regime_label == 'R1' else 0.016
    total_rate = (r.alpha.sum() + r.beta.sum() +
                  r.mu.sum() + r.gamma.sum())
    rate_mkt = total_rate * mkt_fraction * 0.5

    events = []

    for z_idx in range(N_SIZES):
        z = z_idx + 1
        rate = r.mu[z_idx, 0]
        if rate > 1e-8:
            events.append({
                'type': 'sell_can_ask',
                'rate': rate,
                'dz1': -z,
                'dz2': 0,
            })

    for z_idx in range(N_SIZES):
        z = z_idx + 1
        rate = r.gamma[z_idx, 0]
        if rate > 1e-8:
            events.append({
                'type': 'buy_can_bid',
                'rate': rate,
                'dz1': 0,
                'dz2': -z,
            })

    if rate_mkt > 1e-8:
        events.append({
            'type': 'mkt_buy',
            'rate': rate_mkt,
            'dz1': -1,
            'dz2': 0,
        })

    if rate_mkt > 1e-8:
        events.append({
            'type': 'mkt_sell',
            'rate': rate_mkt,
            'dz1': 0,
            'dz2': -1,
        })

    for z_idx in range(N_SIZES):
        z = z_idx + 1
        rate = r.alpha[z_idx, 0]
        if rate > 1e-8:
            events.append({
                'type': 'sell_sub_ask',
                'rate': rate,
                'dz1': +z,
                'dz2': 0,
            })

    for z_idx in range(N_SIZES):
        z = z_idx + 1
        rate = r.beta[z_idx, 0]
        if rate > 1e-8:
            events.append({
                'type': 'buy_sub_bid',
                'rate': rate,
                'dz1': 0,
                'dz2': +z,
            })

    rates_array = np.array([e['rate'] for e in events])
    dz1_array   = np.array([e['dz1']  for e in events])
    dz2_array   = np.array([e['dz2']  for e in events])

    return {
        'events'      : events,
        'rates'       : rates_array,
        'dz1'         : dz1_array,
        'dz2'         : dz2_array,
        'total_rate'  : rates_array.sum(),
        'n_events'    : len(events),
    }

def _simulate_single_state(
    z1_init    : int,
    z2_init    : int,
    r          : RegimeRates,
    rate_struct: dict,
    n_mc       : int,
) -> np.ndarray:
    rates    = rate_struct['rates']
    dz1_arr  = rate_struct['dz1']
    dz2_arr  = rate_struct['dz2']
    n_ev     = rate_struct['n_events']
    outcomes = np.empty(n_mc, dtype=np.float64)

    rng = np.random.default_rng()

    for run in range(n_mc):
        z1 = z1_init
        z2 = z2_init
        t  = 0.0
        price_move = None

        for step in range(MC_MAX_STEPS):
            total_rate = rates.sum()
            if total_rate < 1e-10:
                price_move = 0.5
                break

            dt_event = rng.exponential(1.0 / total_rate)
            t += dt_event

            if t > MC_MAX_TIME:
                eff_ask_up   = rates[dz1_arr < 0].sum() / z1
                eff_ask_down = rates[dz2_arr < 0].sum() / z2
                denom = eff_ask_up + eff_ask_down
                price_move = eff_ask_up / denom if denom > 0 else 0.5
                break

            event_idx = rng.choice(n_ev, p=rates / total_rate)
            dz1 = dz1_arr[event_idx]
            dz2 = dz2_arr[event_idx]

            z1_new = z1 + dz1
            z2_new = z2 + dz2

            if z1_new <= 0:
                price_move = 1
                break
            if z2_new <= 0:
                price_move = 0
                break

            z1 = min(z1_new, N_STATES)
            z2 = min(z2_new, N_STATES)

        if price_move is None:
            price_move = 0.5

        outcomes[run] = price_move

    return outcomes

def compute_empirical_probabilities(
    data    : LOBData,
    verbose : bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if verbose:
        print("\n" + "=" * 60)
        print("EMPIRICAL PROBABILITY COMPUTATION")
        print("=" * 60)
        print("  Scanning data for price move events...")

    df = data.events.copy()

    ask_prices = df['ask1'].values
    bid_prices = df['bid1'].values

    n = len(df)
    next_ask_move = np.zeros(n, dtype=np.int8)

    ask_changes = np.where(np.diff(ask_prices) != 0)[0]

    if verbose:
        print(f"  Total events:        {n:,}")
        print(f"  Ask price changes:   {len(ask_changes):,}")
        print(f"  Change frequency:    "
              f"{len(ask_changes)/n*100:.2f}% of events")

    positions = np.searchsorted(ask_changes, np.arange(n), side='left')

    for i in range(n):
        pos = positions[i]
        if pos < len(ask_changes):
            j = ask_changes[pos]
            if ask_prices[j + 1] > ask_prices[j]:
                next_ask_move[i] = 1
            else:
                next_ask_move[i] = -1

    df['next_ask_move'] = next_ask_move

    df['z1'] = (df['asksz1'] / SHARE_UNIT).round().clip(
        lower=1, upper=N_STATES).astype(int)
    df['z2'] = (df['bidsz1'] / SHARE_UNIT).round().clip(
        lower=1, upper=N_STATES).astype(int)

    df_with_move = df[df['next_ask_move'] != 0].copy()
    df_with_move['ask_up'] = (df_with_move['next_ask_move'] == 1).astype(int)

    if verbose:
        print(f"  Events with observed price move: {len(df_with_move):,}")
        overall_up = df_with_move['ask_up'].mean()
        print(f"  Overall P(ask up): {overall_up:.4f}")
        print(f"  (0.5 = symmetric, >0.5 = upward bias, "
              f"<0.5 = downward bias)")

    emp_R1 = np.full((N_STATES, N_STATES), np.nan)
    emp_R2 = np.full((N_STATES, N_STATES), np.nan)
    n_R1   = np.zeros((N_STATES, N_STATES), dtype=int)
    n_R2   = np.zeros((N_STATES, N_STATES), dtype=int)

    for regime, surface, n_surface in [
        (1, emp_R1, n_R1),
        (2, emp_R2, n_R2),
    ]:
        subset = df_with_move[df_with_move['regime'] == regime]
        grouped = (subset
                   .groupby(['z1', 'z2'])['ask_up']
                   .agg(['mean', 'count'])
                   .reset_index())

        for _, row in grouped.iterrows():
            i = int(row['z1']) - 1
            j = int(row['z2']) - 1
            if 0 <= i < N_STATES and 0 <= j < N_STATES:
                surface[i, j]   = row['mean']
                n_surface[i, j] = int(row['count'])

    if verbose:
        print(f"\n  Empirical surface — R1:")
        _print_surface(emp_R1, label='Empirical R1',
                       n_obs=n_R1)
        print(f"\n  Empirical surface — R2:")
        _print_surface(emp_R2, label='Empirical R2',
                       n_obs=n_R2)

    return emp_R1, emp_R2, n_R1, n_R2

def compare_all(
    kbe_table : ProbabilityTable,
    mc_R1     : np.ndarray,
    mc_R2     : np.ndarray,
    mc_std_R1 : np.ndarray,
    mc_std_R2 : np.ndarray,
    mc_n_R1   : np.ndarray,
    mc_n_R2   : np.ndarray,
    emp_R1    : np.ndarray,
    emp_R2    : np.ndarray,
    emp_n_R1  : np.ndarray,
    emp_n_R2  : np.ndarray,
    verbose   : bool = True,
) -> ValidationResults:
    results = ValidationResults(
        kbe_R1=kbe_table.R1, kbe_R2=kbe_table.R2,
        mc_R1=mc_R1,  mc_R2=mc_R2,
        mc_std_R1=mc_std_R1, mc_std_R2=mc_std_R2,
        mc_n_R1=mc_n_R1, mc_n_R2=mc_n_R2,
        emp_R1=emp_R1, emp_R2=emp_R2,
        emp_n_R1=emp_n_R1, emp_n_R2=emp_n_R2,
    )

    if verbose:
        _print_three_way_table(results, regime=1)
        _print_three_way_table(results, regime=2)
        _print_error_summary(results)

    return results

def _print_three_way_table(
    results: ValidationResults,
    regime : int,
) -> None:
    kbe = results.kbe_R1  if regime == 1 else results.kbe_R2
    mc  = results.mc_R1   if regime == 1 else results.mc_R2
    std = results.mc_std_R1 if regime == 1 else results.mc_std_R2
    emp = results.emp_R1  if regime == 1 else results.emp_R2
    n   = results.emp_n_R1 if regime == 1 else results.emp_n_R2

    label = 'R1 (spread=1 tick)' if regime == 1 else 'R2 (spread≥2 ticks)'
    print(f"\n{'='*80}")
    print(f"TABLE 1 EQUIVALENT — Regime {label}")
    print(f"P(Ask Price Increases) — KBE vs Monte Carlo vs Empirical")
    print(f"{'='*80}")

    print(f"\n{'z1':>4} {'z2':>4} {'KBE':>8} {'MC±std':>14} "
          f"{'Empirical':>12} {'n_emp':>7} "
          f"{'|KBE-MC|':>10} {'|KBE-Emp|':>11}")
    print("-" * 80)

    kbe_mc_errors  = []
    kbe_emp_errors = []

    for i in range(N_STATES):
        for j in range(N_STATES):
            z1, z2 = i + 1, j + 1
            kbe_val = kbe[i, j]
            mc_val  = mc[i, j]
            std_val = std[i, j]
            emp_val = emp[i, j]
            n_val   = n[i, j]

            err_mc  = abs(kbe_val - mc_val)
            err_emp = abs(kbe_val - emp_val) if not np.isnan(emp_val) else np.nan

            kbe_mc_errors.append(err_mc)
            if not np.isnan(err_emp):
                kbe_emp_errors.append(err_emp)

            emp_str = f"{emp_val:.4f}" if not np.isnan(emp_val) else "  N/A "
            err_emp_str = f"{err_emp:.4f}" if not np.isnan(err_emp) else "  N/A "

            flag = ""
            if err_mc > 0.05:
                flag = " ⚠"
            elif err_mc < 0.02:
                flag = " ✓"

            print(f"{z1:>4} {z2:>4} {kbe_val:>8.4f} "
                  f"{mc_val:>7.4f}±{std_val:.4f} "
                  f"{emp_str:>12} {n_val:>7,} "
                  f"{err_mc:>10.4f} {err_emp_str:>11}{flag}")

    print("-" * 80)
    print(f"  Mean |KBE-MC|  error: {np.mean(kbe_mc_errors):.4f}")
    if kbe_emp_errors:
        print(f"  Mean |KBE-Emp| error: {np.mean(kbe_emp_errors):.4f}")
    print(f"  Paper target:         < 0.020 (Table 1 accuracy)")
    quality = ("GOOD" if np.mean(kbe_mc_errors) < 0.05
               else "ACCEPTABLE" if np.mean(kbe_mc_errors) < 0.10
               else "REVIEW NEEDED")
    print(f"  Assessment:           {quality}")

def _print_error_summary(results: ValidationResults) -> None:
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")

    for regime in [1, 2]:
        kbe = results.kbe_R1 if regime == 1 else results.kbe_R2
        mc  = results.mc_R1  if regime == 1 else results.mc_R2
        emp = results.emp_R1 if regime == 1 else results.emp_R2

        kbe_mc_mae  = np.abs(kbe - mc).mean()
        kbe_mc_max  = np.abs(kbe - mc).max()

        valid_emp = ~np.isnan(emp)
        kbe_emp_mae = (np.abs(kbe - emp)[valid_emp].mean()
                       if valid_emp.any() else np.nan)

        print(f"\n  Regime R{regime}:")
        print(f"    KBE vs MC   — MAE: {kbe_mc_mae:.4f}  "
              f"Max: {kbe_mc_max:.4f}")
        print(f"    KBE vs Emp  — MAE: {kbe_emp_mae:.4f}" if not
              np.isnan(kbe_emp_mae) else
              f"    KBE vs Emp  — insufficient data")

    print(f"\n  Key insight:")
    print(f"    KBE-MC disagreement → model specification error")
    print(f"    KBE-Emp disagreement → Poisson/Markov approximation error")
    print(f"    If KBE≈MC but both ≠ Emp → the gap is the Hawkes effect")
    print(f"    This gap motivates the Hawkes extension (future work)")

def plot_validation(
    results  : ValidationResults,
    save_path: Optional[str] = 'figures/validation.png',
) -> None:
    Path('figures').mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'Validation: KBE vs Monte Carlo vs Empirical\n'
        'INTC 2012-06-21 — Spread Regime Comparison',
        fontsize=13, fontweight='bold',
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    for row, regime in enumerate([1, 2]):
        kbe = results.kbe_R1 if regime == 1 else results.kbe_R2
        mc  = results.mc_R1  if regime == 1 else results.mc_R2
        emp = results.emp_R1 if regime == 1 else results.emp_R2
        std = results.mc_std_R1 if regime == 1 else results.mc_std_R2

        label = f"R{regime} (spread={'1' if regime==1 else '≥2'} tick)"

        ax1 = fig.add_subplot(gs[row, 0])
        flat_kbe = kbe.flatten()
        flat_mc  = mc.flatten()
        flat_std = std.flatten()
        ax1.errorbar(flat_kbe, flat_mc, yerr=2*flat_std,
                     fmt='o', alpha=0.7, capsize=3,
                     color='steelblue', markersize=5)
        lims = [0, 1]
        ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect agreement')
        ax1.set_xlabel('KBE P(ask up)')
        ax1.set_ylabel('MC P(ask up)')
        ax1.set_title(f'{label}\nKBE vs Monte Carlo')
        ax1.legend(fontsize=8)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        mae_mc = np.abs(flat_kbe - flat_mc).mean()
        ax1.text(0.05, 0.95, f'MAE={mae_mc:.4f}',
                 transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2 = fig.add_subplot(gs[row, 1])
        valid = ~np.isnan(emp)
        if valid.any():
            ax2.scatter(kbe[valid], emp[valid],
                       alpha=0.7, color='darkgreen', s=40)
            ax2.plot(lims, lims, 'r--', linewidth=1.5,
                    label='Perfect agreement')
            ax2.set_xlabel('KBE P(ask up)')
            ax2.set_ylabel('Empirical P(ask up)')
            ax2.set_title(f'{label}\nKBE vs Empirical')
            ax2.legend(fontsize=8)
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            mae_emp = np.abs(kbe[valid] - emp[valid]).mean()
            ax2.text(0.05, 0.95, f'MAE={mae_emp:.4f}',
                    transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                              facecolor='lightgreen', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'Insufficient empirical data',
                    ha='center', va='center',
                    transform=ax2.transAxes)

        ax3 = fig.add_subplot(gs[row, 2])
        errors_mc = np.abs(flat_kbe - flat_mc)
        ax3.hist(errors_mc, bins=12, alpha=0.7, color='steelblue',
                label=f'KBE-MC errors\nMean={errors_mc.mean():.4f}')
        if valid.any():
            errors_emp = np.abs(kbe[valid] - emp[valid])
            ax3.hist(errors_emp, bins=12, alpha=0.5, color='darkgreen',
                    label=f'KBE-Emp errors\nMean={errors_emp.mean():.4f}')
        ax3.axvline(x=0.02, color='red', linestyle='--',
                   label='Paper target (0.02)')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Count')
        ax3.set_title(f'{label}\nError Distribution')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved validation plot → {save_path}")
    plt.show()

def plot_kbe_vs_mc_heatmap(
    results  : ValidationResults,
    save_path: Optional[str] = 'figures/kbe_vs_mc_heatmap.png',
) -> None:
    Path('figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('KBE vs Monte Carlo — Cell-by-Cell Comparison',
                 fontsize=13, fontweight='bold')

    z_labels = [f'z={z}' for z in range(1, N_STATES+1)]

    for row, regime in enumerate([1, 2]):
        kbe = results.kbe_R1 if regime == 1 else results.kbe_R2
        mc  = results.mc_R1  if regime == 1 else results.mc_R2
        err = np.abs(kbe - mc)

        label = f"R{regime} spread={'1' if regime==1 else '≥2'} tick"

        for col, (data, title, cmap, vmin, vmax) in enumerate([
            (kbe, f'KBE {label}',   'Blues',  0, 1),
            (mc,  f'MC {label}',    'Blues',  0, 1),
            (err, f'|Error| {label}','Reds',   0, None),
        ]):
            ax = axes[row, col]
            vmax_use = vmax if vmax else err.max() + 0.01
            im = ax.imshow(data, cmap=cmap, vmin=vmin,
                          vmax=vmax_use, aspect='auto', origin='lower')
            plt.colorbar(im, ax=ax, shrink=0.8)
            for i in range(N_STATES):
                for j in range(N_STATES):
                    ax.text(j, i, f'{data[i,j]:.3f}',
                           ha='center', va='center', fontsize=7)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Bid Depth z2')
            ax.set_ylabel('Ask Depth z1')
            ax.set_xticks(range(N_STATES))
            ax.set_xticklabels(z_labels, fontsize=7)
            ax.set_yticks(range(N_STATES))
            ax.set_yticklabels(z_labels, fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved KBE vs MC heatmap → {save_path}")
    plt.show()

def _print_surface(
    surface : np.ndarray,
    label   : str,
    std     : Optional[np.ndarray] = None,
    n_obs   : Optional[np.ndarray] = None,
) -> None:
    n = N_STATES
    header = f"  {'z1\\z2':>6}" + "".join(
        f"    z2={j+1}" for j in range(n))
    print(header)
    print("  " + "-" * (8 + 9 * n))
    for i in range(n):
        row_vals = []
        for j in range(n):
            val = surface[i, j]
            if np.isnan(val):
                row_vals.append("  N/A  ")
            elif std is not None:
                row_vals.append(f"{val:.3f}±{std[i,j]:.3f}")
            elif n_obs is not None:
                row_vals.append(f"{val:.3f}({int(n_obs[i,j])})")
            else:
                row_vals.append(f"  {val:.4f}")
        print(f"  z1={i+1:2d}  " + " ".join(row_vals))

if __name__ == '__main__':
    import sys

    MSG_PATH = 'data/raw/intc/intc_messages.csv'
    OB_PATH  = 'data/raw/intc/intc_orderbook.csv'

    if not Path(MSG_PATH).exists():
        print(f"ERROR: {MSG_PATH} not found.")
        sys.exit(1)

    print("Step 1: Loading and calibrating...")
    lob    = load_lobster(MSG_PATH, OB_PATH, verbose=False)
    params = calibrate_fast(lob, verbose=False)

    print("Step 2: Running KBE...")
    kbe_table = run_kbe(params, verbose=False)

    print("\nStep 3: Running Monte Carlo simulation...")
    print(f"  (This takes ~5-15 minutes for {N_MC} reps per state)")
    print(f"  Install tqdm for progress bar: pip install tqdm")
    (mc_R1, mc_R2,
     std_R1, std_R2,
     n_R1, n_R2) = run_monte_carlo(params, n_mc=N_MC, verbose=True)

    print("\nStep 4: Computing empirical probabilities from INTC data...")
    emp_R1, emp_R2, emp_n_R1, emp_n_R2 = compute_empirical_probabilities(
        lob, verbose=True
    )

    print("\nStep 5: Three-way comparison...")
    results = compare_all(
        kbe_table,
        mc_R1, mc_R2, std_R1, std_R2, n_R1, n_R2,
        emp_R1, emp_R2, emp_n_R1, emp_n_R2,
        verbose=True,
    )

    print("\nStep 6: Saving results...")
    Path('results').mkdir(exist_ok=True)
    results.save('results/validation_results.npz')

    df_R1 = results.to_dataframe(regime=1)
    df_R2 = results.to_dataframe(regime=2)
    df_all = pd.concat([df_R1, df_R2], ignore_index=True)
    df_all.to_csv('results/validation_table.csv', index=False)
    print("  Saved validation table → results/validation_table.csv")

    print("\nStep 7: Generating figures...")
    Path('figures').mkdir(exist_ok=True)
    plot_validation(results,          'figures/validation.png')
    plot_kbe_vs_mc_heatmap(results,   'figures/kbe_vs_mc_heatmap.png')

    print("\nvalidation.py completed successfully.")
    print("Next step: run spread_analysis.py")
