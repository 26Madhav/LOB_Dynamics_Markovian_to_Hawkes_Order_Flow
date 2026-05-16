import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

from data_loader import (
    load_lobster, MAX_SIZE_UNITS, MAX_PRICE_LEVELS, SHARE_UNIT
)
from calibration import (
    CalibratedParams, calibrate_fast,
    N_SIZES, N_LEVELS, SIZES, LEVELS,
)

N_STATES   = 6
DT         = 0.0005
T_HORIZON  = 0.2
N_STEPS    = int(T_HORIZON / DT)

@dataclass
class ProbabilityTable:
    R1       : np.ndarray
    R2       : np.ndarray
    diff     : np.ndarray
    z_values : np.ndarray

    def query(self, z1: int, z2: int, regime: int) -> float:
        i = np.clip(z1 - 1, 0, N_STATES - 1)
        j = np.clip(z2 - 1, 0, N_STATES - 1)
        return self.R1[i, j] if regime == 1 else self.R2[i, j]

    def save(self, path: str = 'results/prob_lookup.npz') -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, R1=self.R1, R2=self.R2,
                 diff=self.diff, z_values=self.z_values)
        print(f"  Saved probability table → {path}")

    @classmethod
    def load(cls, path: str = 'results/prob_lookup.npz'):
        d = np.load(path)
        return cls(R1=d['R1'], R2=d['R2'],
                   diff=d['diff'], z_values=d['z_values'])

def build_generator(
    params        : CalibratedParams,
    regime        : int,
    verbose       : bool = True,
) -> np.ndarray:
    r = params.R1 if regime == 1 else params.R2
    n = N_STATES
    total_states = n * n

    def idx(z1: int, z2: int) -> int:
        return (z1 - 1) * n + (z2 - 1)

    L = np.zeros((total_states, total_states))

    rate_sell_sub_at_ask = r.alpha[:, 0].sum()

    rate_buy_sub_at_bid = r.beta[:, 0].sum()

    rate_sell_can_at_ask = r.mu[:, 0].sum()

    rate_buy_can_at_bid = r.gamma[:, 0].sum()

    mkt_fraction_R1 = 0.062
    mkt_fraction_R2 = 0.016
    mkt_fraction = mkt_fraction_R1 if regime == 1 else mkt_fraction_R2

    total_rate = (r.alpha.sum() + r.beta.sum() +
                  r.mu.sum() + r.gamma.sum())
    rate_mkt_buy  = total_rate * mkt_fraction * 0.5
    rate_mkt_sell = total_rate * mkt_fraction * 0.5

    if verbose:
        print(f"\n  Generator rates (Regime R{regime}):")
        print(f"    Sell sub at ask  (z1 ↑): {rate_sell_sub_at_ask:.4f}/s")
        print(f"    Buy  sub at bid  (z2 ↑): {rate_buy_sub_at_bid:.4f}/s")
        print(f"    Sell can at ask  (z1 ↓): {rate_sell_can_at_ask:.4f}/s")
        print(f"    Buy  can at bid  (z2 ↓): {rate_buy_can_at_bid:.4f}/s")
        print(f"    Market buy       (z1 ↓): {rate_mkt_buy:.4f}/s")
        print(f"    Market sell      (z2 ↓): {rate_mkt_sell:.4f}/s")

    def mean_size_for_matrix(matrix_col: np.ndarray) -> float:
        rates = matrix_col
        total = rates.sum()
        if total < 1e-10:
            return 1.0
        sizes = np.arange(1, N_SIZES + 1)
        return (sizes * rates).sum() / total

    mean_z_sell_sub = mean_size_for_matrix(r.alpha[:, 0])
    mean_z_buy_sub  = mean_size_for_matrix(r.beta[:, 0])
    mean_z_sell_can = mean_size_for_matrix(r.mu[:, 0])
    mean_z_buy_can  = mean_size_for_matrix(r.gamma[:, 0])

    dz_sell_sub = max(1, round(mean_z_sell_sub))
    dz_buy_sub  = max(1, round(mean_z_buy_sub))
    dz_sell_can = max(1, round(mean_z_sell_can))
    dz_buy_can  = max(1, round(mean_z_buy_can))

    if verbose:
        print(f"\n  Mean order sizes at best quote:")
        print(f"    Sell submission: {mean_z_sell_sub:.2f} → dz={dz_sell_sub}")
        print(f"    Buy  submission: {mean_z_buy_sub:.2f}  → dz={dz_buy_sub}")
        print(f"    Sell cancel:     {mean_z_sell_can:.2f} → dz={dz_sell_can}")
        print(f"    Buy  cancel:     {mean_z_buy_can:.2f}  → dz={dz_buy_can}")

    for z1 in range(1, n + 1):
        for z2 in range(1, n + 1):
            s = idx(z1, z2)

            z1_new = z1 + dz_sell_sub
            if z1_new <= n:
                s_new = idx(z1_new, z2)
                L[s, s_new] += rate_sell_sub_at_ask

            else:
                pass

            z2_new = z2 + dz_buy_sub
            if z2_new <= n:
                s_new = idx(z1, z2_new)
                L[s, s_new] += rate_buy_sub_at_bid
            else:
                pass

            z1_new = z1 - dz_sell_can
            if z1_new >= 1:
                s_new = idx(z1_new, z2)
                L[s, s_new] += rate_sell_can_at_ask
            else:
                pass

            z1_new = z1 - 1
            if z1_new >= 1:
                s_new = idx(z1_new, z2)
                L[s, s_new] += rate_mkt_buy
            else:
                pass

            z2_new = z2 - dz_buy_can
            if z2_new >= 1:
                s_new = idx(z1, z2_new)
                L[s, s_new] += rate_buy_can_at_bid
            else:
                pass

            z2_new = z2 - 1
            if z2_new >= 1:
                s_new = idx(z1, z2_new)
                L[s, s_new] += rate_mkt_sell
            else:
                pass

    for s in range(total_states):
        L[s, s] = -L[s, :].sum()

    if verbose:
        max_diag = np.abs(np.diag(L)).max()
        stability_number = DT * max_diag
        print(f"\n  Stability check (forward Euler):")
        print(f"    Max |L_ii|          = {max_diag:.4f}")
        print(f"    Δt × max|L_ii|      = {stability_number:.4f}")
        if stability_number < 1.0:
            print(f"    [OK] Stable (< 1.0)")
        else:
            print(f"    [WARN] Potentially unstable! "
                  f"Consider reducing DT = {DT}")

        row_sums = L.sum(axis=1)
        max_row_sum = np.abs(row_sums).max()
        print(f"    Max |row sum|       = {max_row_sum:.6f} "
              f"(should be ~0)")

    return L

def solve_kbe(
    L       : np.ndarray,
    params  : CalibratedParams,
    regime  : int,
    T       : float = T_HORIZON,
    dt      : float = DT,
    verbose : bool  = True,
) -> np.ndarray:
    r = params.R1 if regime == 1 else params.R2
    n = N_STATES
    total_states = n * n
    N = int(T / dt)

    if verbose:
        print(f"\n  Solving KBE (Regime R{regime}):")
        print(f"    Time horizon T = {T}s")
        print(f"    Time step    Δt = {dt}s")
        print(f"    Iterations   N  = {N}")

    mkt_fraction = 0.062 if regime == 1 else 0.016
    total_rate = (r.alpha.sum() + r.beta.sum() +
                  r.mu.sum() + r.gamma.sum())

    rate_ask_depleting = r.mu[:, 0].sum() + total_rate * mkt_fraction * 0.5
    rate_bid_depleting = r.gamma[:, 0].sum() + total_rate * mkt_fraction * 0.5

    if verbose:
        print(f"    Ask-depleting rate: {rate_ask_depleting:.4f}/s")
        print(f"    Bid-depleting rate: {rate_bid_depleting:.4f}/s")

    w = np.zeros(total_states)

    for z1 in range(1, n + 1):
        for z2 in range(1, n + 1):
            s = (z1 - 1) * n + (z2 - 1)

            eff_ask_up   = rate_ask_depleting / z1
            eff_ask_down = rate_bid_depleting / z2
            total_exit   = eff_ask_up + eff_ask_down
            if total_exit > 0:
                w[s] = eff_ask_up / total_exit
            else:
                w[s] = 0.5

    if verbose:
        print(f"    Terminal condition range: "
              f"[{w.min():.4f}, {w.max():.4f}]")

    I = np.eye(total_states)
    propagator = I + dt * L

    max_off = np.abs(propagator - I).max()

    for step in range(N):
        w = propagator @ w

        w = np.clip(w, 0.0, 1.0)

        if verbose and (step + 1) % 100 == 0:
            print(f"    Step {step+1:4d}/{N} — "
                  f"w range: [{w.min():.4f}, {w.max():.4f}]")

    prob_surface = w.reshape(n, n)

    if verbose:
        print(f"\n  P(ask up) surface — Regime R{regime}:")
        print(f"  (rows=ask depth z1, cols=bid depth z2, "
              f"both ∈ {{1,...,{n}}})")
        print()
        header = f"  {'z1\\z2':>6}" + "".join(
            f"  z2={j+1}" for j in range(n))
        print(header)
        print("  " + "-" * (8 + 8 * n))
        for i in range(n):
            row_str = f"  z1={i+1:2d}  " + "".join(
                f"  {prob_surface[i,j]:.4f}" for j in range(n))
            print(row_str)

    return prob_surface

def run_kbe(
    params  : CalibratedParams,
    verbose : bool = True,
) -> ProbabilityTable:
    if verbose:
        print("=" * 60)
        print("KOLMOGOROV BACKWARD EQUATION SOLVER")
        print("=" * 60)

    if verbose:
        print("\n── REGIME R1 (spread = 1 tick) ──────────────────────")
    L_R1   = build_generator(params, regime=1, verbose=verbose)
    surf_R1 = solve_kbe(L_R1, params, regime=1, verbose=verbose)

    if verbose:
        print("\n── REGIME R2 (spread ≥ 2 ticks) ────────────────────")
    L_R2   = build_generator(params, regime=2, verbose=verbose)
    surf_R2 = solve_kbe(L_R2, params, regime=2, verbose=verbose)

    diff = surf_R2 - surf_R1

    if verbose:
        print("\n── DIFFERENCE SURFACE (R2 - R1) ────────────────────")
        print("  Positive = P(ask up) HIGHER when spread widens")
        print("  Negative = P(ask up) LOWER  when spread widens")
        print(f"\n  Max increase: +{diff.max():.4f} "
              f"at state {np.unravel_index(diff.argmax(), diff.shape)}")
        print(f"  Max decrease: {diff.min():.4f} "
              f"at state {np.unravel_index(diff.argmin(), diff.shape)}")
        print(f"  Mean |change|: {np.abs(diff).mean():.4f}")

    table = ProbabilityTable(
        R1       = surf_R1,
        R2       = surf_R2,
        diff     = diff,
        z_values = np.arange(1, N_STATES + 1),
    )

    return table

def plot_probability_surfaces(
    table    : ProbabilityTable,
    save_path: Optional[str] = 'figures/probability_surfaces.png',
) -> None:
    Path('figures').mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        'P(Ask Price Increases at Next Move) — INTC 2012\n'
        'Spread-Regime Conditioned Model '
        '(Extension of Cont-Degond-Xuan 2023)',
        fontsize=13, fontweight='bold',
    )

    z_vals = table.z_values
    configs = [
        (0, table.R1,   'R1: spread = 1 tick',   'Blues',  0, 1),
        (1, table.R2,   'R2: spread ≥ 2 ticks',  'Blues',  0, 1),
        (2, table.diff, 'Difference (R2 - R1)',   'RdBu_r', None, None),
    ]

    axes = []
    for col, surface, title, cmap, vmin, vmax in configs:
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        axes.append(ax)

        Z1, Z2 = np.meshgrid(z_vals, z_vals, indexing='ij')

        if vmin is not None:
            surf = ax.plot_surface(Z1, Z2, surface, cmap=cmap,
                                   vmin=vmin, vmax=vmax,
                                   alpha=0.85, edgecolor='none')
        else:
            abs_max = np.abs(surface).max()
            surf = ax.plot_surface(Z1, Z2, surface, cmap=cmap,
                                   vmin=-abs_max, vmax=abs_max,
                                   alpha=0.85, edgecolor='none')

        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        ax.set_xlabel('Ask Depth z1', fontsize=9)
        ax.set_ylabel('Bid Depth z2', fontsize=9)
        ax.set_zlabel('P(ask up)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(z_vals)
        ax.set_yticks(z_vals)
        ax.view_init(elev=25, azim=225)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved probability surface plot → {save_path}")
    plt.show()

def plot_heatmaps(
    table    : ProbabilityTable,
    save_path: Optional[str] = 'figures/probability_heatmaps.png',
) -> None:
    Path('figures').mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'P(Ask Price Increases) — INTC Spread Regime Comparison\n'
        'Rows = Ask Depth z1, Columns = Bid Depth z2',
        fontsize=12, fontweight='bold',
    )

    z_labels = [f'z={z}' for z in table.z_values]
    abs_max_diff = np.abs(table.diff).max()

    plot_data = [
        (axes[0], table.R1,   'R1: spread=1 tick',   'Blues',
         0, 1),
        (axes[1], table.R2,   'R2: spread≥2 ticks',  'Blues',
         0, 1),
        (axes[2], table.diff, 'Difference (R2 - R1)', 'RdBu_r',
         -abs_max_diff, abs_max_diff),
    ]

    for ax, data, title, cmap, vmin, vmax in plot_data:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax)

        for i in range(N_STATES):
            for j in range(N_STATES):
                val = data[i, j]
                text_color = ('white'
                              if abs(val - (vmin+vmax)/2) > 0.3*(vmax-vmin)
                              else 'black')
                ax.text(j, i, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=8, color=text_color,
                        fontweight='bold')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Bid Depth z2')
        ax.set_ylabel('Ask Depth z1')
        ax.set_xticks(range(N_STATES))
        ax.set_xticklabels(z_labels, fontsize=8)
        ax.set_yticks(range(N_STATES))
        ax.set_yticklabels(z_labels, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved heatmap plot → {save_path}")
    plt.show()

def print_table1_equivalent(table: ProbabilityTable) -> None:
    print("\n" + "=" * 70)
    print("TABLE: P(Ask Up) by Regime and Depth State")
    print("(Equivalent to Table 1 in Cont-Degond-Xuan 2023)")
    print("=" * 70)

    n = N_STATES
    header = f"{'z1 \\ z2':>10}" + "".join(f"   z2={j+1}" for j in range(n))
    print(header)

    for label, surface in [('R1 (s=1)', table.R1),
                            ('R2 (s≥2)', table.R2),
                            ('Diff R2-R1', table.diff)]:
        print(f"\n  {label}:")
        print("  " + "-" * 60)
        for i in range(n):
            row = f"  z1={i+1:2d}     " + "".join(
                f"   {surface[i,j]:+.4f}" for j in range(n))
            print(row)

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

    print("\nStep 2: Running KBE solver...")
    table = run_kbe(params, verbose=True)

    print_table1_equivalent(table)

    print("\nStep 4: Saving probability table...")
    Path('results').mkdir(exist_ok=True)
    table.save('results/prob_lookup.npz')

    print("\nStep 5: Generating figures...")
    Path('figures').mkdir(exist_ok=True)
    plot_heatmaps(table,              'figures/probability_heatmaps.png')
    plot_probability_surfaces(table,  'figures/probability_surfaces.png')

    print("\nkbe_engine.py completed successfully.")
    print("Next step: run validation.py")
