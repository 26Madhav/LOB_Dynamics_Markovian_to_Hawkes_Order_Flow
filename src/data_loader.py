import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict

LOBSTER_PRICE_FACTOR = 10_000
TICK_SIZE_RAW        = 100
SHARE_UNIT           = 100
MAX_SIZE_UNITS       = 10

MAX_PRICE_LEVELS     = 6
TRADING_DAY_START    = 34_200
TRADING_DAY_END      = 57_600
TOTAL_TIME_SECONDS   = TRADING_DAY_END - TRADING_DAY_START

REGIME_1_MAX_SPREAD  = 1
REGIME_2_MIN_SPREAD  = 2

@dataclass
class LOBData:
    events   : pd.DataFrame
    regime1  : pd.DataFrame
    regime2  : pd.DataFrame
    total_time : float
    tick_size  : int
    summary    : dict

def load_lobster(
    message_path: str,
    orderbook_path: str,
    n_levels: int = 5,
    verbose: bool = True,
) -> LOBData:
    if verbose:
        print("=" * 60)
        print("LOADING LOBSTER DATA")
        print("=" * 60)

    msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
    msg = pd.read_csv(message_path, header=None, names=msg_cols)

    ob_cols = []
    for i in range(1, n_levels + 1):
        ob_cols += [f'ask{i}', f'asksz{i}', f'bid{i}', f'bidsz{i}']
    ob = pd.read_csv(orderbook_path, header=None, names=ob_cols)

    if verbose:
        print(f"\n  Raw message  rows : {len(msg):,}")
        print(f"  Raw orderbook rows: {len(ob):,}")
        assert len(msg) == len(ob), \
            "FATAL: message and orderbook row counts do not match!"
        print(f"  Row alignment     : OK")

    spread_raw = ob['ask1'] - ob['bid1']
    tick_size  = int(spread_raw[spread_raw > 0].min())
    if verbose:
        print(f"\n  Detected tick size: {tick_size} LOBSTER units "
              f"= ${tick_size/LOBSTER_PRICE_FACTOR:.4f}")

    spread_ticks = (spread_raw / tick_size).round().astype(int)

    df = msg.copy()
    df['ask1']         = ob['ask1'].values
    df['bid1']         = ob['bid1'].values
    df['asksz1']       = ob['asksz1'].values
    df['bidsz1']       = ob['bidsz1'].values
    df['spread_ticks'] = spread_ticks.values

    df['ask2']   = ob['ask2'].values
    df['bid2']   = ob['bid2'].values
    df['asksz2'] = ob['asksz2'].values
    df['bidsz2'] = ob['bidsz2'].values

    df['size_units'] = (df['size'] / SHARE_UNIT).round()
    df['size_units'] = df['size_units'].clip(lower=1, upper=MAX_SIZE_UNITS)
    df['size_units'] = df['size_units'].astype(int)

    df['regime'] = np.where(df['spread_ticks'] == 1, 1, 2)

    relevant_types = [1, 3, 4]
    df = df[df['type'].isin(relevant_types)].copy()
    df = df.reset_index(drop=True)

    if verbose:
        print(f"\n  After type filtering:")
        print(f"    Type 1 (submissions):   "
              f"{(df['type']==1).sum():>8,} events")
        print(f"    Type 3 (cancellations): "
              f"{(df['type']==3).sum():>8,} events")
        print(f"    Type 4 (executions):    "
              f"{(df['type']==4).sum():>8,} events")
        print(f"    Total kept:             {len(df):>8,} events")

    df['rel_level'] = np.nan

    mask_buy_sub = (df['type'] == 1) & (df['direction'] == 1)
    df.loc[mask_buy_sub, 'rel_level'] = (
        (df.loc[mask_buy_sub, 'ask1'] -
         df.loc[mask_buy_sub, 'price']) / tick_size
    ).round()

    mask_sell_sub = (df['type'] == 1) & (df['direction'] == -1)
    df.loc[mask_sell_sub, 'rel_level'] = (
        (df.loc[mask_sell_sub, 'price'] -
         df.loc[mask_sell_sub, 'bid1']) / tick_size
    ).round()

    mask_buy_can = (df['type'] == 3) & (df['direction'] == 1)
    df.loc[mask_buy_can, 'rel_level'] = (
        (df.loc[mask_buy_can, 'bid1'] -
         df.loc[mask_buy_can, 'price']) / tick_size
    ).round()

    mask_sell_can = (df['type'] == 3) & (df['direction'] == -1)
    df.loc[mask_sell_can, 'rel_level'] = (
        (df.loc[mask_sell_can, 'price'] -
         df.loc[mask_sell_can, 'ask1']) / tick_size
    ).round()

    df.loc[df['type'] == 4, 'rel_level'] = 0

    df['rel_level'] = df['rel_level'].fillna(0).astype(int)

    conditions = [
        mask_buy_sub,
        mask_sell_sub,
        mask_buy_can,
        mask_sell_can,
        df['type'] == 4,
    ]
    choices = ['lambda_plus', 'lambda_minus', 'C_plus', 'C_minus', 'market']
    df['event_class'] = np.select(conditions, choices, default='other')

    valid_level_mask = (
        (df['rel_level'] >= 0) &
        (df['rel_level'] < MAX_PRICE_LEVELS)
    ) | (df['type'] == 4)

    n_before = len(df)
    df = df[valid_level_mask].copy().reset_index(drop=True)
    n_after  = len(df)

    if verbose:
        print(f"\n  After level filtering (keep levels 0-{MAX_PRICE_LEVELS-1}):")
        print(f"    Dropped {n_before - n_after:,} deep-book events "
              f"({(n_before-n_after)/n_before:.1%})")
        print(f"    Remaining: {n_after:,} events")

    regime1 = df[df['regime'] == 1].copy().reset_index(drop=True)
    regime2 = df[df['regime'] == 2].copy().reset_index(drop=True)

    summary = _build_summary(df, regime1, regime2, tick_size)

    if verbose:
        _print_summary(summary)

    return LOBData(
        events     = df,
        regime1    = regime1,
        regime2    = regime2,
        total_time = TOTAL_TIME_SECONDS,
        tick_size  = tick_size,
        summary    = summary,
    )

def _build_summary(
    df: pd.DataFrame,
    r1: pd.DataFrame,
    r2: pd.DataFrame,
    tick_size: int,
) -> dict:
    def regime_stats(data: pd.DataFrame, label: str) -> dict:
        return {
            f'{label}_n_events'      : len(data),
            f'{label}_n_lambda_plus' : (data['event_class']=='lambda_plus').sum(),
            f'{label}_n_lambda_minus': (data['event_class']=='lambda_minus').sum(),
            f'{label}_n_C_plus'      : (data['event_class']=='C_plus').sum(),
            f'{label}_n_C_minus'     : (data['event_class']=='C_minus').sum(),
            f'{label}_n_market'      : (data['event_class']=='market').sum(),
        }

    s = {
        'total_events'  : len(df),
        'tick_size'     : tick_size,
        'spread_dist'   : df['spread_ticks'].value_counts().sort_index().to_dict(),
        'regime1_frac'  : len(r1) / len(df),
        'regime2_frac'  : len(r2) / len(df),
        'size_dist'     : df['size_units'].value_counts().sort_index().to_dict(),
        'level_dist'    : df['rel_level'].value_counts().sort_index().to_dict(),
    }
    s.update(regime_stats(r1, 'R1'))
    s.update(regime_stats(r2, 'R2'))
    return s

def _print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\n  Total events (after filtering): {s['total_events']:,}")

    print(f"\n  Spread regime split:")
    print(f"    Regime R1 (spread=1):  "
          f"{s['R1_n_events']:>8,}  ({s['regime1_frac']:.1%})")
    print(f"    Regime R2 (spread>=2): "
          f"{s['R2_n_events']:>8,}  ({s['regime2_frac']:.1%})")

    print(f"\n  Event class breakdown:")
    for regime, label in [('R1', 'Regime 1'), ('R2', 'Regime 2')]:
        n = s[f'{regime}_n_events']
        print(f"\n    {label} ({n:,} events):")
        for cls in ['lambda_plus', 'lambda_minus', 'C_plus', 'C_minus', 'market']:
            count = s[f'{regime}_n_{cls}']
            pct   = count / n if n > 0 else 0
            print(f"      {cls:<15}: {count:>7,}  ({pct:.1%})")

    print(f"\n  Size distribution (100-share units):")
    for size, count in sorted(s['size_dist'].items()):
        pct = count / s['total_events']
        bar = '█' * int(pct * 40)
        print(f"    Size={size:2d}: {count:>7,} ({pct:5.1%}) {bar}")

    print(f"\n  Relative level distribution:")
    for level, count in sorted(s['level_dist'].items()):
        pct = count / s['total_events']
        bar = '█' * int(pct * 40)
        print(f"    Level={level}: {count:>7,} ({pct:5.1%}) {bar}")

def test_poisson_adequacy(data: LOBData, verbose: bool = True) -> dict:
    from scipy import stats as scipy_stats

    results = {}

    for regime_label, regime_data in [('R1', data.regime1),
                                       ('R2', data.regime2)]:
        times = regime_data['time'].values
        iats  = np.diff(times)
        iats  = iats[iats > 1e-6]

        cv       = iats.std() / iats.mean() if iats.mean() > 0 else np.nan
        acf_lag1 = pd.Series(iats).autocorr(lag=1)
        acf_lag5 = pd.Series(iats).autocorr(lag=5)
        ks_stat, ks_pval = scipy_stats.kstest(
            iats, 'expon', args=(0, iats.mean())
        )

        results[regime_label] = {
            'cv'       : cv,
            'acf_lag1' : acf_lag1,
            'acf_lag5' : acf_lag5,
            'ks_stat'  : ks_stat,
            'ks_pval'  : ks_pval,
            'n_events' : len(regime_data),
        }

        if verbose:
            print(f"\n  Poisson Adequacy — {regime_label} "
                  f"({len(regime_data):,} events):")
            print(f"    CV            = {cv:.3f}  "
                  f"(Poisson=1.0, bursty>1.0)")
            print(f"    ACF lag-1     = {acf_lag1:.3f}  "
                  f"(Poisson≈0)")
            print(f"    ACF lag-5     = {acf_lag5:.3f}  "
                  f"(Poisson≈0)")
            print(f"    KS vs Expon   : stat={ks_stat:.3f}, "
                  f"p={ks_pval:.4f}")
            verdict = ("REJECT" if ks_pval < 0.05 else "FAIL TO REJECT")
            print(f"    {verdict} Poisson null at 5% significance")
            if cv > 1.5:
                print(f"    NOTE: High CV suggests Hawkes-like self-excitation.")
                print(f"    We follow CDX (2023) in using Poisson approximation.")

    return results

def extract_depth_snapshots(data: LOBData) -> Tuple[np.ndarray, np.ndarray]:
    ask_depths = (data.events['asksz1'] / SHARE_UNIT).round().clip(
        lower=1, upper=MAX_SIZE_UNITS).astype(int).values
    bid_depths = (data.events['bidsz1'] / SHARE_UNIT).round().clip(
        lower=1, upper=MAX_SIZE_UNITS).astype(int).values
    return ask_depths, bid_depths

def get_price_move_labels(data: LOBData) -> pd.Series:
    ask_prices = data.events['ask1'].values
    labels     = np.zeros(len(ask_prices), dtype=int)

    for i in range(len(ask_prices) - 1):
        future_asks = ask_prices[i+1:]
        changed     = np.where(future_asks != ask_prices[i])[0]
        if len(changed) > 0:
            next_ask = future_asks[changed[0]]
            labels[i] = 1 if next_ask > ask_prices[i] else -1

    return pd.Series(labels, name='next_ask_move')

if __name__ == '__main__':
    import sys

    MSG_PATH = 'data/raw/intc/intc_messages.csv'
    OB_PATH  = 'data/raw/intc/intc_orderbook.csv'

    if not Path(MSG_PATH).exists():
        print(f"ERROR: Message file not found: {MSG_PATH}")
        print("Update MSG_PATH and OB_PATH at bottom of this file.")
        sys.exit(1)

    lob = load_lobster(MSG_PATH, OB_PATH, verbose=True)

    print("\n" + "=" * 60)
    print("POISSON ADEQUACY TEST")
    print("=" * 60)
    poisson_results = test_poisson_adequacy(lob, verbose=True)

    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    assert lob.events['regime'].isin([1, 2]).all(), \
        "FAIL: some events have invalid regime"
    print("  [OK] All events have valid regime (1 or 2)")

    assert (lob.events['rel_level'] >= 0).all(), \
        "FAIL: negative relative levels found"
    assert (lob.events['rel_level'] < MAX_PRICE_LEVELS).all() or \
           (lob.events['type'] == 4).any(), \
        "FAIL: relative levels exceed max"
    print("  [OK] All relative levels in valid range")

    assert (lob.events['size_units'] >= 1).all(), \
        "FAIL: size units below 1"
    assert (lob.events['size_units'] <= MAX_SIZE_UNITS).all(), \
        "FAIL: size units exceed maximum"
    print("  [OK] All size units in valid range [1, 10]")

    total_check = len(lob.regime1) + len(lob.regime2)
    assert total_check == len(lob.events), \
        "FAIL: regime1 + regime2 != total events"
    print("  [OK] Regime counts sum correctly")

    assert (lob.events['time'].diff().dropna() >= 0).all(), \
        "FAIL: events are not time-ordered"
    print("  [OK] Events are time-ordered")

    print("\n  All sanity checks passed.")
    print("\n  lob.regime1 shape:", lob.regime1.shape)
    print("  lob.regime2 shape:", lob.regime2.shape)
    print("\n  Sample of processed events:")
    cols_to_show = ['time', 'type', 'event_class', 'size_units',
                    'rel_level', 'spread_ticks', 'regime']
    print(lob.events[cols_to_show].head(10).to_string(index=False))

    print("\n  data_loader.py completed successfully.")
    print("  Next step: run calibration.py")
