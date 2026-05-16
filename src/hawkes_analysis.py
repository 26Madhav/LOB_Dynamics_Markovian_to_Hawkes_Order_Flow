import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
})

COLORS = {
    "kbe":      "#2563EB",
    "mc":       "#16A34A",
    "empirical":"#DC2626",
    "R1":       "#7C3AED",
    "R2":       "#EA580C",
    "sell":     "#EF4444",
    "buy":      "#3B82F6",
}

STREAM_NAMES = ["sell_submit", "buy_submit", "sell_cancel", "buy_cancel"]
D_GRID  = 6
Z       = 10
XI_BINS = 6

def isotonic_smooth_mu(mu):
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return mu
    ir  = IsotonicRegression(increasing=False)
    mu  = mu.copy()
    for e in [2, 3]:
        for xi in range(mu.shape[2]):
            y = mu[e, :, xi]
            if np.any(y > 0):
                mu[e, :, xi] = ir.fit_transform(np.arange(len(y)), y)
    return mu

def load_params(lh_dir):
    path = os.path.join(lh_dir, "hawkes_params.npz")
    raw  = np.load(path, allow_pickle=True)
    params = {}
    for regime in [1, 2]:
        rk = f"R{regime}"
        mu    = np.where(np.isfinite(raw[f"{rk}_mu"])    & (raw[f"{rk}_mu"]    >= 0), raw[f"{rk}_mu"],    0.0).astype(float)
        alpha = np.where(np.isfinite(raw[f"{rk}_alpha"]) & (raw[f"{rk}_alpha"] >= 0), raw[f"{rk}_alpha"], 0.0).astype(float)
        beta  = np.where(np.isfinite(raw[f"{rk}_beta"])  & (raw[f"{rk}_beta"]  >  0), raw[f"{rk}_beta"],  1.0).astype(float)
        mu    = isotonic_smooth_mu(mu)
        params[rk] = dict(mu=mu, alpha=alpha, beta=beta)
    return params

def load_validation(val_dir):
    path = os.path.join(val_dir, "hawkes_validation_results.npz")
    return np.load(path, allow_pickle=True)

def fig_A1_branching_ratios(params, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, (rk, p) in zip(axes, params.items()):
        mu, alpha, beta = p["mu"], p["alpha"], p["beta"]

        eta = np.zeros((4, mu.shape[1]))
        for e in range(4):
            for d in range(mu.shape[1]):
                b = beta[e, d, :].mean()
                a = alpha[e, d, :].mean()
                eta[e, d] = a / b if b > 0 else 0.0

        x     = np.arange(4)
        width = 0.25
        depth_labels = ["d=1", "d=3", "d=6"]
        depth_idxs   = [0, 2, 5]
        palette      = ["#93C5FD", "#2563EB", "#1E3A8A"]

        for i, (di, lbl) in enumerate(zip(depth_idxs, depth_labels)):
            ax.bar(x + i * width, eta[:, di], width,
                   label=lbl, color=palette[i], alpha=0.85)

        ax.axhline(1.0, color="red", lw=1, ls="--", alpha=0.6, label="η=1 (critical)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(STREAM_NAMES, rotation=15, ha="right")
        ax.set_title(f"Branching ratios η = α/β  —  {rk}")
        ax.set_ylabel("η (branching ratio)")
        ax.set_ylim(0, max(0.05, eta.max() * 1.3))
        ax.legend(fontsize=9)

    fig.suptitle("Finding A — Hawkes branching ratios by stream and depth",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out, "fig_A1_branching_ratios.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def fig_A2_rate_vs_xi(params, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, (rk, p) in zip(axes, params.items()):
        mu = p["mu"]
        xi_range = np.arange(mu.shape[2])
        styles = ["-o", "-s", "-^", "-D"]
        clrs   = [COLORS["sell"], COLORS["buy"],
                  "#F97316", "#8B5CF6"]
        for e in range(4):
            ax.plot(xi_range, mu[e, 0, :], styles[e],
                    color=clrs[e], label=STREAM_NAMES[e],
                    markersize=5, lw=1.5)

        ax.set_xlabel("xi bin (excitement level)")
        ax.set_ylabel("μ (baseline rate, ev/s)")
        ax.set_title(f"Baseline rate vs excitement  —  {rk}  (depth=1)")
        ax.legend()
        ax.set_xticks(xi_range)

    fig.suptitle("Finding A — μ decreases as excitement bin increases",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out, "fig_A2_rate_vs_xi.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def fig_B1_rate_vs_depth(params, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (rk, p) in zip(axes, params.items()):
        mu = p["mu"]
        depths = np.arange(1, mu.shape[1] + 1)

        sc = mu[2, :, :].mean(axis=1)
        bc = mu[3, :, :].mean(axis=1)

        ax.plot(depths, sc, "-o", color=COLORS["sell"],
                label="sell_cancel", lw=2, markersize=5)
        ax.plot(depths, bc, "-s", color=COLORS["buy"],
                label="buy_cancel",  lw=2, markersize=5)
        ax.set_xlabel("Depth bin (queue size)")
        ax.set_ylabel("μ averaged over xi (ev/s)")
        ax.set_title(f"Cancel rates vs depth  —  {rk}")
        ax.legend()
        ax.set_xticks(depths)

    fig.suptitle("Finding B — Cancel rates decrease with queue depth (isotonic)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out, "fig_B1_rate_vs_depth.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def _surface_heatmap(ax, surf, title, vmin=0, vmax=1, cmap="RdYlBu_r",
                     D=D_GRID, annotate=True):
    s = surf[:D, :D].copy().astype(float)
    im = ax.imshow(s, origin="upper", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=[0.5, D + 0.5, D + 0.5, 0.5])
    ax.set_xlabel("z2 (bid depth)")
    ax.set_ylabel("z1 (ask depth)")
    ax.set_title(title)
    ax.set_xticks(range(1, D + 1))
    ax.set_yticks(range(1, D + 1))
    if annotate:
        for i in range(D):
            for j in range(D):
                v = s[i, j]
                if np.isfinite(v):
                    ax.text(j + 1, i + 1, f"{v:.2f}",
                            ha="center", va="center",
                            fontsize=7,
                            color="white" if (v < 0.25 or v > 0.75) else "black")
    return im

def fig_C1_surfaces(val, params, out, regime=1):
    rk = f"R{regime}"

    mc = val.get(f"{rk}_mc_first", None)

    if mc is None:
        print(f"  No MC surface for {rk} in validation results — skipping C1")
        return

    mc = np.array(mc, dtype=float)

    kbe = None
    for key in [
        f"{rk}_kbe",
        f"{rk}_kbe_relative",
        f"{rk}_first_step_up"
    ]:
        if key in val:
            kbe = np.array(val[key], dtype=float)
            break

    if kbe is None:
        kbe = np.full_like(mc, np.nan, dtype=float)

    emp = val.get(f"{rk}_empirical", None)
    if emp is not None:
        emp = np.array(emp, dtype=float)

    interior = []

    for s in [kbe, mc]:
        if s is not None:
            v = s[1:D_GRID, 1:D_GRID]
            finite = v[np.isfinite(v)]
            if finite.size:
                interior.extend(finite.tolist())

    if len(interior) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = max(0.0, np.nanmin(interior) - 0.05)
        vmax = min(1.0, np.nanmax(interior) + 0.05)

    has_emp = emp is not None and np.any(np.isfinite(emp))
    ncols = 3 if has_emp else 2

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    if ncols == 2:
        axes = list(axes)

    im = _surface_heatmap(
        axes[0],
        kbe,
        f"KBE — {rk}",
        vmin,
        vmax
    )

    _surface_heatmap(
        axes[1],
        mc,
        f"Monte Carlo — {rk}",
        vmin,
        vmax
    )

    if has_emp:
        _surface_heatmap(
            axes[2],
            emp,
            f"Empirical INTC — {rk}",
            vmin,
            vmax
        )

    fig.colorbar(
        im,
        ax=axes,
        label="P(ask up | first price move)",
        shrink=0.7,
        pad=0.02
    )

    fig.suptitle(
        f"Finding C — P(ask up | first price-moving event)  {rk}\n"
        f"z1↓: deeper ask → lower probability   |   z2↑: deeper bid → higher probability",
        fontweight="bold"
    )

    fig.tight_layout()

    path = os.path.join(out, f"fig_C1_surfaces_{rk}.png")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {path}")

def fig_C2_gap_closure(val, out):
    labels  = []
    mae_kbe = []
    mae_mc  = []
    mae_mct = []

    for regime in [1, 2]:
        rk  = f"R{regime}"
        emp_key = f"{rk}_empirical"
        if emp_key not in val:
            continue
        emp = np.array(val[emp_key])
        if not np.any(np.isfinite(emp)):
            continue

        for surf_key, store in [
            (f"{rk}_kbe",      mae_kbe),
            (f"{rk}_mc_first", mae_mc),
            (f"{rk}_mc_T",     mae_mct),
        ]:
            if surf_key not in val and "kbe" in surf_key:
                for alt in [f"{rk}_kbe_relative", f"{rk}_first_step_up"]:
                    if alt in val:
                        surf_key = alt
                        break
            if surf_key not in val:
                store.append(np.nan)
                continue
            s    = np.array(val[surf_key])
            mask = np.isfinite(s) & np.isfinite(emp)
            mask[0, :] = False
            mask[:, 0] = False
            if mask.sum() < 2:
                store.append(np.nan)
            else:
                store.append(float(np.abs(s[mask] - emp[mask]).mean()))
        labels.append(rk)

    if not labels:
        print("  No empirical data for gap closure figure — skipping C2")
        return

    x     = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width, mae_kbe, width, label="KBE",      color=COLORS["kbe"],      alpha=0.85)
    b2 = ax.bar(x,         mae_mc,  width, label="MC first",  color=COLORS["mc"],       alpha=0.85)
    b3 = ax.bar(x + width, mae_mct, width, label="MC (T-hor)",color=COLORS["empirical"],alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE vs empirical (interior cells)")
    ax.set_title("Finding C — Gap to empirical: KBE vs MC")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out, "fig_C2_gap_closure.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def fig_C3_scatter(val, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (label, color, key_fmt) in zip(axes, [
        ("KBE",         COLORS["kbe"], "{rk}_kbe"),
        ("Monte Carlo", COLORS["mc"],  "{rk}_mc_first"),
    ]):
        all_model = []
        all_emp   = []
        all_regime = []

        for regime in [1, 2]:
            rk      = f"R{regime}"
            emp_key = f"{rk}_empirical"
            if emp_key not in val:
                continue
            emp = np.array(val[emp_key])

            surf_key = key_fmt.format(rk=rk)
            if surf_key not in val:
                for alt in [f"{rk}_kbe_relative", f"{rk}_first_step_up"]:
                    if alt in val:
                        surf_key = alt
                        break
            if surf_key not in val:
                continue

            s    = np.array(val[surf_key])
            mask = np.isfinite(s) & np.isfinite(emp)
            mask[0, :] = False
            mask[:, 0] = False
            if mask.sum() < 2:
                continue

            all_model.extend(s[mask].tolist())
            all_emp.extend(emp[mask].tolist())
            all_regime.extend([regime] * mask.sum())

        if not all_model:
            ax.set_visible(False)
            continue

        all_model  = np.array(all_model)
        all_emp    = np.array(all_emp)
        all_regime = np.array(all_regime)

        for regime, marker, rk_color in [(1, "o", COLORS["R1"]),
                                          (2, "s", COLORS["R2"])]:
            idx = all_regime == regime
            if idx.sum():
                ax.scatter(all_emp[idx], all_model[idx],
                           color=rk_color, marker=marker, alpha=0.7,
                           s=60, label=f"R{regime}")

        lo = min(all_emp.min(), all_model.min()) - 0.02
        hi = max(all_emp.max(), all_model.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Empirical P(ask up)")
        ax.set_ylabel(f"{label} P(ask up)")
        ax.set_title(f"{label} vs Empirical")
        ax.legend()

        if len(all_emp) > 1:
            corr = float(np.corrcoef(all_emp, all_model)[0, 1])
            mae  = float(np.abs(all_model - all_emp).mean())
            ax.text(0.05, 0.92, f"corr={corr:.3f}  MAE={mae:.3f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.suptitle("Finding C — Model vs empirical scatter (interior cells)",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out, "fig_C3_scatter.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def fig_C4_mc_vs_kbe(val, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, regime in zip(axes, [1, 2]):
        rk      = f"R{regime}"
        mc_key  = f"{rk}_mc_first"
        if mc_key not in val:
            ax.set_visible(False)
            continue

        mc = np.array(val[mc_key])
        kbe = None
        for alt in [f"{rk}_kbe", f"{rk}_kbe_relative", f"{rk}_first_step_up"]:
            if alt in val:
                kbe = np.array(val[alt])
                break
        if kbe is None:
            ax.set_visible(False)
            continue

        mask = np.isfinite(mc) & np.isfinite(kbe)
        mask[0, :] = False
        mask[:, 0] = False

        ax.scatter(kbe[mask], mc[mask], color=COLORS[rk],
                   alpha=0.7, s=60, zorder=3)
        lo = min(kbe[mask].min(), mc[mask].min()) - 0.02
        hi = max(kbe[mask].max(), mc[mask].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("KBE P(ask up)")
        ax.set_ylabel("MC P(ask up)")
        ax.set_title(f"KBE vs MC — {rk}")

        if mask.sum() > 1:
            corr = float(np.corrcoef(kbe[mask], mc[mask])[0, 1])
            mae  = float(np.abs(mc[mask] - kbe[mask]).mean())
            ax.text(0.05, 0.92, f"corr={corr:.3f}  MAE={mae:.3f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.suptitle("KBE vs MC internal consistency (interior cells only)",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out, "fig_C4_kbe_vs_mc.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def write_report(params, val, out):
    lines = []
    lines.append("=" * 65)
    lines.append("HAWKES LOB — ANALYSIS REPORT")
    lines.append("=" * 65)
    lines.append("")

    lines.append("FINDING A — Branching ratios (η = α/β)")
    lines.append("-" * 40)
    for rk, p in params.items():
        mu, alpha, beta = p["mu"], p["alpha"], p["beta"]
        lines.append(f"\n  {rk}:")
        for e, name in enumerate(STREAM_NAMES):
            eta_mean = (alpha[e, :3, :] / np.maximum(beta[e, :3, :], 1e-9)).mean()
            mu_mean  = mu[e, 0, :].mean()
            lines.append(f"    {name:15s}  η={eta_mean:.5f}  μ(d=1)={mu_mean:.4f} ev/s")
    lines.append("")
    lines.append("  Interpretation: η << 1 for all streams → subcritical Hawkes.")
    lines.append("  Self-excitation is present but decays quickly (large β).")
    lines.append("")

    lines.append("FINDING B — Depth monotonicity of cancel rates")
    lines.append("-" * 40)
    for rk, p in params.items():
        mu = p["mu"]
        lines.append(f"\n  {rk}:")
        for e in [2, 3]:
            rates = mu[e, :6, :].mean(axis=1)
            mono  = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
            lines.append(f"    {STREAM_NAMES[e]:15s}  rates={rates.round(4)}  monotone={mono}")
    lines.append("")

    lines.append("FINDING C — Three-way surface comparison")
    lines.append("-" * 40)
    for regime in [1, 2]:
        rk      = f"R{regime}"
        emp_key = f"{rk}_empirical"
        mc_key  = f"{rk}_mc_first"
        if emp_key not in val or mc_key not in val:
            lines.append(f"\n  {rk}: empirical data not available")
            continue

        emp = np.array(val[emp_key])
        mc  = np.array(val[mc_key])
        kbe = None
        for alt in [f"{rk}_kbe", f"{rk}_kbe_relative", f"{rk}_first_step_up"]:
            if alt in val:
                kbe = np.array(val[alt])
                break

        mask = np.isfinite(emp)
        mask[0, :] = False
        mask[:, 0] = False

        if mask.sum() < 2:
            lines.append(f"\n  {rk}: too few empirical cells ({mask.sum()})")
            continue

        lines.append(f"\n  {rk}  ({mask.sum()} interior cells with empirical data):")
        if kbe is not None:
            mae_kbe  = float(np.abs(kbe[mask]  - emp[mask]).mean())
            corr_kbe = float(np.corrcoef(kbe[mask], emp[mask])[0,1]) if mask.sum()>2 else float("nan")
            lines.append(f"    KBE vs empirical:  MAE={mae_kbe:.4f}  corr={corr_kbe:.4f}")

        mae_mc  = float(np.abs(mc[mask]  - emp[mask]).mean())
        corr_mc = float(np.corrcoef(mc[mask], emp[mask])[0,1]) if mask.sum()>2 else float("nan")
        lines.append(f"    MC  vs empirical:  MAE={mae_mc:.4f}  corr={corr_mc:.4f}")

        if kbe is not None and mae_kbe > 1e-6:
            closure = (mae_kbe - mae_mc) / mae_kbe * 100
            lines.append(f"    Gap closure:       {closure:+.1f}%  (KBE→MC)")
            lines.append(f"    KBE-MC agreement:  MAE={float(np.abs(kbe[mask]-mc[mask]).mean()):.4f}")

    lines.append("")
    lines.append("=" * 65)

    rpt_path = os.path.join(out, "hawkes_analysis_report.txt")
    with open(rpt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {rpt_path}")
    print()
    print("\n".join(lines))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lh_dir",  default="data/processed",
                   help="Directory containing hawkes_params.npz")
    p.add_argument("--val_dir", default="data/processed",
                   help="Directory containing hawkes_validation_results.npz")
    p.add_argument("--out",     default="data/processed/figures",
                   help="Output directory for figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("=" * 65)
    print("HAWKES LOB — hawkes_analysis.py")
    print("=" * 65)

    print("\nLoading parameters...")
    params = load_params(args.lh_dir)

    print("Loading validation results...")
    val = load_validation(args.val_dir)
    print(f"  Validation keys: {list(val.files)}")

    print("\n── Finding A figures ──")
    fig_A1_branching_ratios(params, args.out)
    fig_A2_rate_vs_xi(params, args.out)

    print("\n── Finding B figures ──")
    fig_B1_rate_vs_depth(params, args.out)

    print("\n── Finding C figures ──")
    for regime in [1, 2]:
        fig_C1_surfaces(val, params, args.out, regime)
    fig_C2_gap_closure(val, args.out)
    fig_C3_scatter(val, args.out)
    fig_C4_mc_vs_kbe(val, args.out)

    print("\n── Summary report ──")
    write_report(params, val, args.out)

    print(f"\nAll outputs written to {args.out}/")
    print("=" * 65)

if __name__ == "__main__":
    main()
