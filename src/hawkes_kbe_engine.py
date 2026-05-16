import argparse
import os
import time

import numpy as np
import scipy.sparse as sp

Z        = 10
XI_BINS  = 6
N_TYPES  = 4
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

def encode(z1: int, z2: int, xi: int) -> int:
    return (z1 - 1) * Z * XI_BINS + (z2 - 1) * XI_BINS + xi

def decode(idx: int):
    xi = idx % XI_BINS
    z2 = (idx // XI_BINS) % Z + 1
    z1 = idx // (Z * XI_BINS) + 1
    return z1, z2, xi

N_STATES = Z * Z * XI_BINS

def depth_idx(e_type: int, z1: int, z2: int, max_d: int) -> int:
    raw = (z1 - 1) if e_type in (0, 2) else (z2 - 1)
    return min(raw, max_d - 1)

def get_rate(mu, alpha, beta, e_type: int,
             z1: int, z2: int, xi: int) -> float:
    xi_idx = min(xi, XI_BINS - 1)
    d_idx  = depth_idx(e_type, z1, z2, mu.shape[1])

    return float(mu[e_type, d_idx, xi_idx])

def build_generator(mu, alpha, beta) -> sp.csr_matrix:
    rows, cols, vals = [], [], []

    for z1 in range(1, Z + 1):
        for z2 in range(1, Z + 1):
            for xi in range(XI_BINS):
                s = encode(z1, z2, xi)
                total_out = 0.0

                for e_type in range(N_TYPES):
                    rate = get_rate(
                        mu, alpha, beta,
                        e_type, z1, z2, xi
                    )

                    if rate <= 0:
                        continue

                    xi_new = min(xi + 1, XI_BINS - 1)

                    if e_type == 0:
                        z1n, z2n = min(z1 + 1, Z), z2

                    elif e_type == 1:
                        z1n, z2n = z1, min(z2 + 1, Z)

                    elif e_type == 2:
                        z1n, z2n = max(z1 - 1, 1), z2

                    else:
                        z1n, z2n = z1, max(z2 - 1, 1)

                    t = encode(z1n, z2n, xi_new)

                    if t != s:
                        rows.append(s)
                        cols.append(t)
                        vals.append(rate)
                        total_out += rate

                if total_out > 0:
                    rows.append(s)
                    cols.append(s)
                    vals.append(-total_out)

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(N_STATES, N_STATES)
    )

def first_step_ask_up(mu, alpha, beta) -> np.ndarray:
    surf = np.full((Z, Z), np.nan)

    for z1 in range(1, Z + 1):
        for z2 in range(1, Z + 1):
            if z1 == 1 and z2 == 1:
                continue

            vals = []

            for xi in range(XI_BINS):
                r_ask_up = (
                    get_rate(mu, alpha, beta, 2, z1, z2, xi)
                    if z1 > 1 else 0.0
                )

                r_bid_up = (
                    get_rate(mu, alpha, beta, 3, z1, z2, xi)
                    if z2 > 1 else 0.0
                )

                r_move = r_ask_up + r_bid_up

                if r_move > 0:
                    vals.append(r_ask_up / r_move)

            if vals:
                surf[z1 - 1, z2 - 1] = float(np.mean(vals))

    return surf

def kbe_ask_up_T(mu, alpha, beta,
                 T: float,
                 dt: float,
                 verbose: bool = True) -> np.ndarray:
    L = build_generator(mu, alpha, beta)

    N = L.shape[0]

    diag_max = abs(float(L.diagonal().min()))

    if diag_max * dt >= 1.0:
        dt = 0.95 / diag_max

        if verbose:
            print(f"    dt adjusted to {dt:.6f}")

    NT   = max(1, int(round(T / dt)))
    dt   = T / NT

    I    = sp.eye(N, format="csr")
    step = I + dt * L

    if verbose:
        print(f"    KBE: N={N}  T={T}s  dt={dt:.6f}  steps={NT}")

    surf = np.zeros((Z, Z))

    t0 = time.time()

    for z1_0 in range(1, Z + 1):
        for z2_0 in range(1, Z + 1):
            f = np.array([
                1.0 if decode(idx)[0] < z1_0 else 0.0
                for idx in range(N)
            ])

            w = f.copy()

            for _ in range(NT):
                w = step.dot(w)
                np.clip(w, 0.0, 1.0, out=w)

            xi_vals = [
                w[encode(z1_0, z2_0, xi)]
                for xi in range(XI_BINS)
            ]

            surf[z1_0 - 1, z2_0 - 1] = float(np.mean(xi_vals))

    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s")

    return surf

def print_surface(surf: np.ndarray,
                  label: str,
                  D: int = D_GRID):
    print(f"\n  {label}:")
    print("         " + "".join(f"  z2={j+1}" for j in range(D)))

    for i in range(D):
        row = "  z1={:d}   ".format(i + 1)

        for j in range(D):
            v = surf[i, j]

            row += (
                f"  {'  nan' if np.isnan(v) else f'{v:.4f}'}"
            )

        print(row)

    v = surf[1:D, 1:D]
    valid = v[np.isfinite(v)]

    if len(valid):
        print(
            f"  min={valid.min():.4f}  "
            f"max={valid.max():.4f}  "
            f"mean={valid.mean():.4f}  "
            f"(interior only)"
        )

    z1_ok, z1_tot = 0, 0
    z2_ok, z2_tot = 0, 0

    for j in range(1, D):
        col = surf[1:D, j]
        fin = np.isfinite(col)

        if fin.sum() >= 2:
            z1_ok += int(
                col[fin][-1] <= col[fin][0]
            )

            z1_tot += 1

    for i in range(1, D):
        row = surf[i, 1:D]
        fin = np.isfinite(row)

        if fin.sum() >= 2:
            z2_ok += int(
                row[fin][-1] >= row[fin][0]
            )

            z2_tot += 1

    print(
        f"  z1↓ monotone (interior): "
        f"{z1_ok}/{z1_tot} cols"
    )

    print(
        f"  z2↑ monotone (interior): "
        f"{z2_ok}/{z2_tot} rows"
    )

def main():
    p = argparse.ArgumentParser(
        description="Hawkes LOB — KBE engine v4"
    )

    p.add_argument("--lh_dir", default="data/processed")
    p.add_argument("--out",    default="data/processed")

    p.add_argument("--T",  type=float, default=0.2)
    p.add_argument("--dt", type=float, default=0.0005)

    p.add_argument(
        "--M",
        type=int,
        default=6,
        help="XI_BINS override"
    )

    p.add_argument(
        "--method",
        default="first_step",
        choices=["first_step", "kbe_T", "both"]
    )

    args = p.parse_args()

    global XI_BINS
    XI_BINS = args.M

    os.makedirs(args.out, exist_ok=True)

    print("=" * 65)
    print("HAWKES LOB — hawkes_kbe_engine.py")
    print("=" * 65)

    print(f"  Z={Z}")
    print(f"  XI_BINS={XI_BINS}")
    print(f"  N_STATES={Z*Z*XI_BINS}")

    params_path = os.path.join(
        args.lh_dir,
        "hawkes_params.npz"
    )

    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)

    raw = np.load(params_path, allow_pickle=True)

    results = {}

    for regime in [1, 2]:
        rk = f"R{regime}"

        print(f"\n{'='*65}")
        print(f"  Regime {rk}")
        print(f"{'='*65}")

        mu = np.where(
            np.isfinite(raw[f"{rk}_mu"]) &
            (raw[f"{rk}_mu"] >= 0),
            raw[f"{rk}_mu"],
            0.0
        ).astype(float)
        mu = isotonic_smooth_mu(mu)

        alpha = np.where(
            np.isfinite(raw[f"{rk}_alpha"]) &
            (raw[f"{rk}_alpha"] >= 0),
            raw[f"{rk}_alpha"],
            0.0
        ).astype(float)

        beta = np.where(
            np.isfinite(raw[f"{rk}_beta"]) &
            (raw[f"{rk}_beta"] > 0),
            raw[f"{rk}_beta"],
            1.0
        ).astype(float)

        print(f"  mu    ∈ [{mu.min():.4g}, {mu.max():.4g}]")
        print(f"  alpha ∈ [{alpha.min():.4g}, {alpha.max():.4g}]")
        print(f"  beta  ∈ [{beta.min():.4g}, {beta.max():.4g}]")

        print(f"\n  Rate check at (z1=3, z2=3, xi=3):")

        names = [
            "sell_submit",
            "buy_submit ",
            "sell_cancel",
            "buy_cancel "
        ]

        for e in range(N_TYPES):
            r = get_rate(
                mu, alpha, beta,
                e, 3, 3, 3
            )

            print(f"    {names[e]}: {r:.5f} ev/s")

        if args.method in ("first_step", "both"):
            print(
                "\n  [Method 1] "
                "Analytic first-step "
                "P(ask up | first price move)"
            )

            t0 = time.time()

            surf = first_step_ask_up(
                mu, alpha, beta
            )

            print(
                f"  Computed in "
                f"{time.time()-t0:.3f}s"
            )

            print_surface(
                surf,
                f"P(ask up | first move) — {rk}"
            )

            results[f"{rk}_first_step_up"] = surf
            results[f"{rk}_kbe_relative"] = surf

        if args.method in ("kbe_T", "both"):
            print(
                f"\n  [Method 2] "
                f"KBE P(ask queue thins within T={args.T}s)"
            )

            surf_T = kbe_ask_up_T(
                mu, alpha, beta,
                args.T,
                args.dt,
                verbose=True
            )

            print_surface(
                surf_T,
                f"P(z1 < z1_0 within T={args.T}s) — {rk}"
            )

            results[f"{rk}_kbe_T"] = surf_T

    results.update(
        z1_vals=np.arange(1, Z + 1),
        z2_vals=np.arange(1, Z + 1),
        T=np.array(args.T),
        M=np.array(XI_BINS),
    )

    npz_out = os.path.join(
        args.out,
        "hawkes_kbe_results.npz"
    )

    np.savez(npz_out, **results)

    print(f"\nSaved {npz_out}")

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)

    for regime in [1, 2]:
        rk = f"R{regime}"

        if f"{rk}_first_step_up" in results:
            s = results[f"{rk}_first_step_up"]

            v = s[1:D_GRID, 1:D_GRID]
            valid = v[np.isfinite(v)]

            print(f"\n  {rk} first_step_up surface:")

            if len(valid):
                print(
                    f"    mean={valid.mean():.4f}  "
                    f"min={valid.min():.4f}  "
                    f"max={valid.max():.4f}"
                )

    print("=" * 65)

if __name__ == "__main__":
    main()
