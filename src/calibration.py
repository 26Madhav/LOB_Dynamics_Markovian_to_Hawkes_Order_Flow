import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from data_loader import (
    load_lobster, LOBData,
    MAX_SIZE_UNITS, MAX_PRICE_LEVELS,
    TRADING_DAY_START, TRADING_DAY_END,
)

SIZES  = np.arange(1, MAX_SIZE_UNITS + 1)
LEVELS = np.arange(0, MAX_PRICE_LEVELS)
N_SIZES  = len(SIZES)
N_LEVELS = len(LEVELS)

@dataclass
class RegimeRates:
    alpha  : np.ndarray
    beta   : np.ndarray
    mu     : np.ndarray
    gamma  : np.ndarray
    T      : float
    n_events     : int
    regime_label : str

    def to_dict(self) -> dict:
        return {
            'alpha' : self.alpha,
            'beta'  : self.beta,
            'mu'    : self.mu,
            'gamma' : self.gamma,
            'T'     : self.T,
            'n_events'     : self.n_events,
            'regime_label' : self.regime_label,
        }

@dataclass
class CalibratedParams:
    R1 : RegimeRates
    R2 : RegimeRates

    def save(self, path: str = 'results/calibrated_params.npz') -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,

            R1_alpha = self.R1.alpha,
            R1_beta  = self.R1.beta,
            R1_mu    = self.R1.mu,
            R1_gamma = self.R1.gamma,
            R1_T     = self.R1.T,

            R2_alpha = self.R2.alpha,
            R2_beta  = self.R2.beta,
            R2_mu    = self.R2.mu,
            R2_gamma = self.R2.gamma,
            R2_T     = self.R2.T,
        )
        print(f"  Saved calibrated parameters → {path}")

    @classmethod
    def load(cls, path: str = 'results/calibrated_params.npz'):
        d = np.load(path)
        R1 = RegimeRates(
            alpha=d['R1_alpha'], beta=d['R1_beta'],
            mu=d['R1_mu'],       gamma=d['R1_gamma'],
            T=float(d['R1_T']),  n_events=0,
            regime_label='R1',
        )
        R2 = RegimeRates(
            alpha=d['R2_alpha'], beta=d['R2_beta'],
            mu=d['R2_mu'],       gamma=d['R2_gamma'],
            T=float(d['R2_T']),  n_events=0,
            regime_label='R2',
        )
        return cls(R1=R1, R2=R2)

def calibrate(data: LOBData, verbose: bool = True) -> CalibratedParams:
    if verbose:
        print("=" * 60)
        print("CALIBRATING RATE MATRICES")
        print("=" * 60)

    R1_rates = _calibrate_regime(
        data.regime1,
        regime_label='R1',
        total_day_time=data.total_time,
        regime_fraction=len(data.regime1) / len(data.events),
        verbose=verbose,
    )

    R2_rates = _calibrate_regime(
        data.regime2,
        regime_label='R2',
        total_day_time=data.total_time,
        regime_fraction=len(data.regime2) / len(data.events),
        verbose=verbose,
    )

    params = CalibratedParams(R1=R1_rates, R2=R2_rates)

    if verbose:
        _print_comparison(params)

    return params

def _calibrate_regime(
    regime_data   : pd.DataFrame,
    regime_label  : str,
    total_day_time: float,
    regime_fraction: float,
    verbose       : bool = True,
) -> RegimeRates:
    T = total_day_time * regime_fraction

    if verbose:
        print(f"\n  Regime {regime_label}:")
        print(f"    Events in regime : {len(regime_data):,}")
        print(f"    Regime fraction  : {regime_fraction:.3f}")
        print(f"    Time in regime   : {T:.1f} seconds")

    alpha = np.zeros((N_SIZES, N_LEVELS))
    beta  = np.zeros((N_SIZES, N_LEVELS))
    mu    = np.zeros((N_SIZES, N_LEVELS))
    gamma = np.zeros((N_SIZES, N_LEVELS))

    class_to_matrix = {
        'lambda_minus': alpha,
        'lambda_plus' : beta,
        'C_minus'     : mu,
        'C_plus'      : gamma,
    }

    for event_class, matrix in class_to_matrix.items():
        subset = regime_data[regime_data['event_class'] == event_class]

        for _, row in subset.iterrows():
            z = int(row['size_units']) - 1
            k = int(row['rel_level'])
            if 0 <= z < N_SIZES and 0 <= k < N_LEVELS:
                matrix[z, k] += 1

        matrix /= T if T > 0 else 1.0

    if verbose:
        _print_regime_rates(alpha, beta, mu, gamma, regime_label)

    return RegimeRates(
        alpha=alpha, beta=beta, mu=mu, gamma=gamma,
        T=T, n_events=len(regime_data),
        regime_label=regime_label,
    )

def _calibrate_regime_fast(
    regime_data   : pd.DataFrame,
    regime_label  : str,
    total_day_time: float,
    regime_fraction: float,
    verbose       : bool = True,
) -> RegimeRates:
    T = total_day_time * regime_fraction

    alpha = np.zeros((N_SIZES, N_LEVELS))
    beta  = np.zeros((N_SIZES, N_LEVELS))
    mu    = np.zeros((N_SIZES, N_LEVELS))
    gamma = np.zeros((N_SIZES, N_LEVELS))

    class_to_matrix = {
        'lambda_minus': alpha,
        'lambda_plus' : beta,
        'C_minus'     : mu,
        'C_plus'      : gamma,
    }

    for event_class, matrix in class_to_matrix.items():
        subset = regime_data[regime_data['event_class'] == event_class].copy()
        if len(subset) == 0:
            continue

        subset['z_idx'] = (subset['size_units'] - 1).clip(0, N_SIZES - 1)
        subset['k_idx'] = subset['rel_level'].clip(0, N_LEVELS - 1)

        counts = (subset
                  .groupby(['z_idx', 'k_idx'])
                  .size()
                  .reset_index(name='count'))

        for _, row in counts.iterrows():
            z = int(row['z_idx'])
            k = int(row['k_idx'])
            matrix[z, k] = row['count'] / T

    if verbose:
        print(f"\n  Regime {regime_label}: {len(regime_data):,} events, "
              f"T={T:.1f}s")
        _print_regime_rates(alpha, beta, mu, gamma, regime_label)

    return RegimeRates(
        alpha=alpha, beta=beta, mu=mu, gamma=gamma,
        T=T, n_events=len(regime_data),
        regime_label=regime_label,
    )

def calibrate_fast(data: LOBData, verbose: bool = True) -> CalibratedParams:
    if verbose:
        print("=" * 60)
        print("CALIBRATING RATE MATRICES (vectorised)")
        print("=" * 60)

    total_events = len(data.events)

    R1_rates = _calibrate_regime_fast(
        data.regime1,
        regime_label    = 'R1',
        total_day_time  = data.total_time,
        regime_fraction = len(data.regime1) / total_events,
        verbose         = verbose,
    )

    R2_rates = _calibrate_regime_fast(
        data.regime2,
        regime_label    = 'R2',
        total_day_time  = data.total_time,
        regime_fraction = len(data.regime2) / total_events,
        verbose         = verbose,
    )

    params = CalibratedParams(R1=R1_rates, R2=R2_rates)

    if verbose:
        _print_comparison(params)

    return params

def validate_rates(params: CalibratedParams, verbose: bool = True) -> bool:
    if verbose:
        print("\n" + "=" * 60)
        print("RATE MATRIX VALIDATION")
        print("=" * 60)

    all_passed = True

    for label, regime in [('R1', params.R1), ('R2', params.R2)]:
        for name, mat in [('alpha', regime.alpha), ('beta', regime.beta),
                          ('mu', regime.mu), ('gamma', regime.gamma)]:
            if (mat < 0).any():
                print(f"  [FAIL] {label}.{name} has negative rates!")
                all_passed = False
    if all_passed:
        print("  [OK] All rates non-negative")

    for label, regime in [('R1', params.R1), ('R2', params.R2)]:
        for name, mat in [('alpha', regime.alpha), ('beta', regime.beta)]:
            col0 = mat[:, 0].sum()
            col5 = mat[:, 5].sum()
            if col0 < col5 * 0.5:
                print(f"  [WARN] {label}.{name}: level-0 rate < level-5 rate "
                      f"(expected decay). col0={col0:.4f}, col5={col5:.4f}")
    print("  [OK] Level decay pattern checked")

    for label, regime in [('R1', params.R1), ('R2', params.R2)]:
        for name, mat in [('beta', regime.beta), ('alpha', regime.alpha)]:
            sz1 = mat[0, :].sum()
            sz10 = mat[9, :].sum()
            if sz1 < sz10:
                print(f"  [WARN] {label}.{name}: size=1 rate < size=10 rate "
                      f"(note: size=10 includes truncated large orders)")
    print("  [OK] Size decay pattern checked (size=10 truncation noted)")

    r1_total = (params.R1.alpha.sum() + params.R1.beta.sum() +
                params.R1.mu.sum()   + params.R1.gamma.sum())
    r2_total = (params.R2.alpha.sum() + params.R2.beta.sum() +
                params.R2.mu.sum()   + params.R2.gamma.sum())
    if verbose:
        print(f"\n  Total rate comparison:")
        print(f"    R1 total rate: {r1_total:.4f} events/second")
        print(f"    R2 total rate: {r2_total:.4f} events/second")
        print(f"    Ratio R1/R2  : {r1_total/r2_total:.2f}x")

    if verbose:
        print(f"\n  Most active cells (size=1):")
        print(f"    {'Matrix':<12} {'Level-0':>10} {'Level-1':>10} "
              f"{'Level-2':>10} {'Level-5':>10}")
        print(f"    {'-'*52}")
        for label, regime in [('R1', params.R1), ('R2', params.R2)]:
            for name, mat in [('beta', regime.beta),
                               ('alpha', regime.alpha)]:
                row = mat[0, :]
                print(f"    {label+'.'+name:<12} {row[0]:>10.4f} "
                      f"{row[1]:>10.4f} {row[2]:>10.4f} {row[5]:>10.4f}")

    return all_passed

def analyze_spread_effect(params: CalibratedParams,
                           verbose: bool = True) -> dict:
    if verbose:
        print("\n" + "=" * 60)
        print("SPREAD EFFECT ON RATE MATRICES")
        print("=" * 60)
        print("  (How do order flow rates change when spread widens?)")

    results = {}

    matrix_pairs = [
        ('beta',  params.R1.beta,  params.R2.beta,  'Buy submissions λ+'),
        ('alpha', params.R1.alpha, params.R2.alpha, 'Sell submissions λ-'),
        ('gamma', params.R1.gamma, params.R2.gamma, 'Buy cancels C+'),
        ('mu',    params.R1.mu,    params.R2.mu,    'Sell cancels C-'),
    ]

    for name, r1_mat, r2_mat, label in matrix_pairs:
        r1_total = r1_mat.sum()
        r2_total = r2_mat.sum()
        pct_change = ((r2_total - r1_total) / r1_total * 100
                      if r1_total > 0 else np.nan)

        r1_by_level = r1_mat.sum(axis=0)
        r2_by_level = r2_mat.sum(axis=0)
        level_changes = np.where(
            r1_by_level > 0,
            (r2_by_level - r1_by_level) / r1_by_level * 100,
            np.nan,
        )

        results[name] = {
            'r1_total'     : r1_total,
            'r2_total'     : r2_total,
            'pct_change'   : pct_change,
            'level_changes': level_changes,
        }

        if verbose:
            direction = "↑" if pct_change > 0 else "↓"
            print(f"\n  {label}:")
            print(f"    R1 total rate: {r1_total:.4f}/s")
            print(f"    R2 total rate: {r2_total:.4f}/s")
            print(f"    Change: {direction} {abs(pct_change):.1f}%")
            print(f"    By level: ", end='')
            for k, chg in enumerate(level_changes):
                if not np.isnan(chg):
                    d = "↑" if chg > 0 else "↓"
                    print(f"L{k}:{d}{abs(chg):.0f}%  ", end='')
            print()

    if verbose:
        print("\n  KEY FINDING:")
        print("  Market orders (executions) change tells us about")
        print("  informed trading intensity at different spread levels:")
        r1_mkt = (data_regime_market_rate(params.R1))
        r2_mkt = (data_regime_market_rate(params.R2))
        if r1_mkt > 0:
            print(f"    Market order rate R1: {r1_mkt:.4f}/s")
            print(f"    Market order rate R2: {r2_mkt:.4f}/s")

    return results

def data_regime_market_rate(regime: RegimeRates) -> float:
    return 0.0

def _print_regime_rates(
    alpha: np.ndarray, beta: np.ndarray,
    mu: np.ndarray,    gamma: np.ndarray,
    regime_label: str,
) -> None:
    print(f"\n    Rate matrix totals ({regime_label}):")
    print(f"      β  (buy  submissions λ+): "
          f"{beta.sum():>8.4f} events/s")
    print(f"      α  (sell submissions λ-): "
          f"{alpha.sum():>8.4f} events/s")
    print(f"      γ  (buy  cancels    C+): "
          f"{gamma.sum():>8.4f} events/s")
    print(f"      μ  (sell cancels    C-): "
          f"{mu.sum():>8.4f} events/s")
    print(f"      Total: "
          f"{(alpha+beta+mu+gamma).sum():>8.4f} events/s")

    print(f"\n    β (buy submissions) — size=1 row, by level:")
    header = "      " + "".join(f"  L{k}" for k in range(N_LEVELS))
    print(header)
    row_str = "      " + "".join(f"{beta[0,k]:>5.3f}" for k in range(N_LEVELS))
    print(row_str)

def _print_comparison(params: CalibratedParams) -> None:
    print("\n" + "=" * 60)
    print("REGIME COMPARISON (R1 vs R2)")
    print("=" * 60)

    print(f"\n  {'Matrix':<25} {'R1 total':>12} {'R2 total':>12} "
          f"{'R2/R1':>8}")
    print(f"  {'-'*60}")

    for name, r1_mat, r2_mat, label in [
        ('β (buy  submit)', params.R1.beta,  params.R2.beta,  None),
        ('α (sell submit)', params.R1.alpha, params.R2.alpha, None),
        ('γ (buy  cancel)', params.R1.gamma, params.R2.gamma, None),
        ('μ (sell cancel)', params.R1.mu,    params.R2.mu,    None),
    ]:
        r1 = r1_mat.sum()
        r2 = r2_mat.sum()
        ratio = r2 / r1 if r1 > 0 else np.nan
        print(f"  {name:<25} {r1:>12.4f} {r2:>12.4f} {ratio:>8.3f}x")

    print(f"\n  Interpretation:")
    print(f"  Ratio > 1 means the rate INCREASES when spread widens.")
    print(f"  Ratio < 1 means the rate DECREASES when spread widens.")
    print(f"  This is your core microstructure finding.")

def plot_rate_matrices(
    params   : CalibratedParams,
    save_path: Optional[str] = 'figures/rate_matrices.png',
) -> None:
    Path('figures').mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Rate Matrices: INTC Spread-Regime Comparison\n'
                 '(Cont-Degond-Xuan Framework, Extended)',
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.35)

    vmax_beta  = max(params.R1.beta.max(),  params.R2.beta.max())
    vmax_alpha = max(params.R1.alpha.max(), params.R2.alpha.max())

    plot_configs = [

        (0, 0, params.R1.beta,               'β R1 (Buy Submit, spread=1)',
         vmax_beta,  'Blues'),
        (0, 1, params.R2.beta,               'β R2 (Buy Submit, spread≥2)',
         vmax_beta,  'Blues'),
        (0, 2, params.R2.beta-params.R1.beta,'β Difference (R2 - R1)',
         None,       'RdBu_r'),
        (1, 0, params.R1.alpha,              'α R1 (Sell Submit, spread=1)',
         vmax_alpha, 'Reds'),
        (1, 1, params.R2.alpha,              'α R2 (Sell Submit, spread≥2)',
         vmax_alpha, 'Reds'),
        (1, 2, params.R2.alpha-params.R1.alpha,'α Difference (R2 - R1)',
         None,        'RdBu_r'),
    ]

    for row, col, matrix, title, vmax, cmap in plot_configs:
        ax = fig.add_subplot(gs[row, col])

        if vmax is not None:
            im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                           vmin=0, vmax=vmax)
        else:
            abs_max = np.abs(matrix).max()
            im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                           vmin=-abs_max, vmax=abs_max)

        plt.colorbar(im, ax=ax, shrink=0.8, label='Rate (events/s)')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Price Level k (from best quote)')
        ax.set_ylabel('Order Size z (100-share units)')
        ax.set_xticks(range(N_LEVELS))
        ax.set_xticklabels([f'L{k}' for k in range(N_LEVELS)])
        ax.set_yticks(range(N_SIZES))
        ax.set_yticklabels([f'z={z}' for z in SIZES])

        for i in range(N_SIZES):
            for j in range(N_LEVELS):
                val = matrix[i, j]
                text = f'{val:.3f}' if abs(val) >= 0.001 else ''
                color = 'white' if abs(val) > 0.5 * (vmax or 1) else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=6, color=color)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved rate matrix plot → {save_path}")
    plt.show()

def plot_level_profiles(
    params   : CalibratedParams,
    save_path: Optional[str] = 'figures/level_profiles.png',
) -> None:
    Path('figures').mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Order Flow Rate Profiles by Price Level\n'
                 'INTC 2012-06-21 — Spread Regime Comparison',
                 fontsize=13, fontweight='bold')

    configs = [
        (axes[0,0], params.R1.beta,  params.R2.beta,
         'β: Buy Submission Rate λ+', 'blue'),
        (axes[0,1], params.R1.alpha, params.R2.alpha,
         'α: Sell Submission Rate λ-', 'red'),
        (axes[1,0], params.R1.gamma, params.R2.gamma,
         'γ: Buy Cancellation Rate C+', 'darkblue'),
        (axes[1,1], params.R1.mu,    params.R2.mu,
         'μ: Sell Cancellation Rate C-', 'darkred'),
    ]

    levels = np.arange(N_LEVELS)

    for ax, r1_mat, r2_mat, title, color in configs:
        r1_by_level = r1_mat.sum(axis=0)
        r2_by_level = r2_mat.sum(axis=0)

        ax.plot(levels, r1_by_level, 'o-', color=color,
                label='R1 (spread=1)', linewidth=2, markersize=7)
        ax.plot(levels, r2_by_level, 's--', color=color, alpha=0.6,
                label='R2 (spread≥2)', linewidth=2, markersize=7)

        ax.fill_between(levels, r1_by_level, r2_by_level,
                        alpha=0.15, color=color,
                        label='Regime difference')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Relative Price Level k')
        ax.set_ylabel('Total Rate (events/s, summed over sizes)')
        ax.set_xticks(levels)
        ax.set_xticklabels([f'L{k}' for k in levels])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved level profile plot → {save_path}")
    plt.show()

if __name__ == '__main__':
    import sys
    from pathlib import Path

    MSG_PATH = 'data/raw/intc/intc_messages.csv'
    OB_PATH  = 'data/raw/intc/intc_orderbook.csv'

    if not Path(MSG_PATH).exists():
        print(f"ERROR: {MSG_PATH} not found.")
        sys.exit(1)

    print("Step 1: Loading data...")
    lob = load_lobster(MSG_PATH, OB_PATH, verbose=False)

    print("\nStep 2: Calibrating rate matrices...")
    params = calibrate_fast(lob, verbose=True)

    print("\nStep 3: Validating...")
    validate_rates(params, verbose=True)

    print("\nStep 4: Saving...")
    Path('results').mkdir(exist_ok=True)
    params.save('results/calibrated_params.npz')

    print("\nStep 5: Spread effect analysis...")
    analyze_spread_effect(params, verbose=True)

    print("\nStep 6: Generating plots...")
    Path('figures').mkdir(exist_ok=True)
    plot_level_profiles(params, save_path='figures/level_profiles.png')
    plot_rate_matrices(params,  save_path='figures/rate_matrices.png')

    print("\ncalibration.py completed successfully.")
    print("Next step: run kbe_engine.py")
