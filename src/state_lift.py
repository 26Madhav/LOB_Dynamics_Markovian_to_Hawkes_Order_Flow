import argparse
import os
import time

import numpy as np
import scipy.sparse as sp

Z      = 10
K      = 6
N_TYPES = 4

TYPE_LABELS = ["sell_submit", "buy_submit", "sell_cancel", "buy_cancel"]

def encode(z1: int, z2: int, k: int, xi: int, M: int) -> int:
    return ((z1 - 1) * Z + (z2 - 1)) * K * M + k * M + xi

def decode(idx: int, M: int) -> tuple:
    xi  = idx % M
    k   = (idx // M) % K
    z2  = (idx // (K * M)) % Z + 1
    z1  = idx // (Z * K * M) + 1
    return z1, z2, k, xi

def _verify_encoding(M: int):
    for z1, z2, k, xi in [(1, 1, 0, 0), (Z, Z, K-1, M-1), (3, 7, 2, 1)]:
        idx = encode(z1, z2, k, xi, M)
        z1r, z2r, kr, xir = decode(idx, M)
        assert (z1r, z2r, kr, xir) == (z1, z2, k, xi), \
            f"Encoding round-trip failed for ({z1},{z2},{k},{xi})"

def lob_transition(z1: int, z2: int, k: int,
                   event_type: int, size: int) -> tuple:
    if event_type == 0:
        z1p = min(z1 + size, Z)
        z2p = z2
    elif event_type == 1:
        z1p = z1
        z2p = min(z2 + size, Z)
    elif event_type == 2:
        z1p = max(z1 - size, 1)
        z2p = z2
    else:
        z1p = z1
        z2p = max(z2 - size, 1)

    total  = z1p + z2p
    kp = min(int((total - 2) * K // (2 * Z)), K - 1)

    return z1p, z2p, kp

def build_intensity_grid(mu_arr: np.ndarray, alpha_arr: np.ndarray,
                         beta_arr: np.ndarray, M: int) -> tuple:
    mask = alpha_arr > 1e-10

    if mask.sum() == 0:
        return np.zeros(M), 1.0

    mu_med    = np.median(mu_arr[mask])
    alpha_med = np.median(alpha_arr[mask])
    beta_med  = np.median(beta_arr[mask])
    n_med     = alpha_med / beta_med
    n_med     = min(n_med, 0.999)

    E_X = mu_med / (beta_med * (1.0 - n_med))

    x_grid   = np.arange(M) * E_X
    beta_eff = beta_med

    return x_grid, beta_eff

def build_generator(mu_arr: np.ndarray,
                    alpha_arr: np.ndarray,
                    beta_arr: np.ndarray,
                    M: int,
                    verbose: bool = True) -> sp.csr_matrix:
    N = Z * Z * K * M
    _verify_encoding(M)

    x_grid, beta_eff = build_intensity_grid(mu_arr, alpha_arr, beta_arr, M)

    if verbose:
        print(f"    N = {N} states,  M = {M}")
        print(f"    x_grid     = {x_grid}")
        print(f"    beta_eff   = {beta_eff:.4f} s^-1")

    rows, cols, data = [], [], []

    def add(r, c, v):
        if r != c and abs(v) > 0.0:
            rows.append(r)
            cols.append(c)
            data.append(v)

    t0 = time.time()

    for z1 in range(1, Z + 1):
        for z2 in range(1, Z + 1):
            for k in range(K):
                for xi in range(M):
                    s = encode(z1, z2, k, xi, M)
                    total_rate_out = 0.0

                    for ti in range(N_TYPES):
                        for zi in range(1, Z + 1):
                            ki = k

                            mu_i    = mu_arr   [ti, zi - 1, ki]
                            alpha_i = alpha_arr[ti, zi - 1, ki]

                            lam_i = mu_i + alpha_i * x_grid[xi]

                            if lam_i <= 0.0:
                                continue

                            z1p, z2p, kp = lob_transition(z1, z2, k, ti, zi)

                            xip = min(xi + 1, M - 1)

                            sp_idx = encode(z1p, z2p, kp, xip, M)

                            if sp_idx == s:
                                continue

                            add(s, sp_idx, lam_i)
                            total_rate_out += lam_i

                    if xi > 0:
                        decay_rate = beta_eff * xi
                        xid        = xi - 1
                        sd         = encode(z1, z2, k, xid, M)
                        add(s, sd, decay_rate)
                        total_rate_out += decay_rate

                    rows.append(s)
                    cols.append(s)
                    data.append(-total_rate_out)

    elapsed = time.time() - t0
    if verbose:
        print(f"    Built in {elapsed:.1f}s  |  nnz (raw) = {len(data):,}")

    L = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    row_sums = np.array(L.sum(axis=1)).flatten()
    max_err  = np.abs(row_sums).max()
    if verbose:
        print(f"    Row-sum max error: {max_err:.2e}  (should be < 1e-10)")
    assert max_err < 1e-6, f"Generator row-sum check failed: max |err| = {max_err:.2e}"

    return L

def save_sparse(L: sp.csr_matrix, path: str):
    np.savez(path,
             data    = L.data,
             indices = L.indices,
             indptr  = L.indptr,
             shape   = np.array(L.shape))
    print(f"    Saved -> {path}.npz  (nnz={L.nnz:,})")

def load_sparse(path: str) -> sp.csr_matrix:
    d = np.load(path + ".npz")
    return sp.csr_matrix(
        (d["data"], d["indices"], d["indptr"]),
        shape=tuple(d["shape"])
    )

def summarise(L: sp.csr_matrix, label: str, M: int) -> str:
    N = L.shape[0]
    row_sums = np.array(L.sum(axis=1)).flatten()
    diag     = np.array(L.diagonal())

    lines = [
        f"\n{'='*55}",
        f"  {label}",
        f"{'='*55}",
        f"  States N        = {N}  (Z={Z}, K={K}, M={M})",
        f"  nnz             = {L.nnz:,}",
        f"  Density         = {L.nnz / N**2 * 100:.4f}%",
        f"  Row-sum max err = {np.abs(row_sums).max():.2e}",
        f"  Diagonal range  : [{diag.min():.4f}, {diag.max():.4f}]",
        f"  Max exit rate   : {-diag.min():.4f} events/s",
        f"  Mean exit rate  : {-diag.mean():.4f} events/s",
    ]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Hawkes LOB — state lift  (build L_H generator matrix)"
    )
    parser.add_argument(
        "--params",
        default="data/processed/hawkes_params.npz",
        help="Path to hawkes_params.npz"
    )
    parser.add_argument("--out", default="data/processed",
                        help="Output directory")
    parser.add_argument("--M",   type=int, default=4,
                        help="Number of intensity discretisation levels (default 4)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    M = args.M

    print("=" * 55)
    print("HAWKES LOB — state_lift.py")
    print("=" * 55)

    print("\nLoading Hawkes parameters...")
    p = np.load(args.params, allow_pickle=False)
    print(f"  Loaded keys: {list(p.keys())}")

    summary_lines = []

    for regime in [1, 2]:
        print(f"\nBuilding L_H for regime R{regime}...")
        mu_arr    = p[f"R{regime}_mu"]
        alpha_arr = p[f"R{regime}_alpha"]
        beta_arr  = p[f"R{regime}_beta"]

        print(f"  Parameter shapes: mu={mu_arr.shape}  "
              f"alpha={alpha_arr.shape}  beta={beta_arr.shape}")
        print(f"  Non-zero alpha (Hawkes) streams: "
              f"{(alpha_arr > 1e-10).sum()} / {alpha_arr.size}")

        L = build_generator(mu_arr, alpha_arr, beta_arr, M, verbose=True)

        out_path = os.path.join(args.out, f"L_H_R{regime}")
        save_sparse(L, out_path)

        summary_lines.append(summarise(L, f"L_H  Regime {regime}", M))

    summary_path = os.path.join(args.out, "state_lift_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved state_lift_summary.txt -> {summary_path}")

    for line in summary_lines:
        print(line)

    print("\n" + "=" * 55)
    print("state_lift.py complete.")
    print("=" * 55)

if __name__ == "__main__":
    main()
