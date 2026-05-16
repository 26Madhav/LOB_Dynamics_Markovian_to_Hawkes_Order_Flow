# Order Book Dynamics — Project README

---

## What is this project about?

This project is based on the research paper **OrderBook_dynamics.pdf** by Cont, Degond, and Xuan. The paper builds a mathematical framework for how a limit order book (LOB) evolves over time. The main idea is simple: orders come in (the order flow), and then the book gets updated (market clearing). If the order flow behaves like a Markov process, you can write down a generator for the whole system and use the **Backward Kolmogorov Equation (KBE)** to compute probabilities like:

> *Given the current state of the book, what is the probability that the ask price goes up next?*

The project first reproduces that exact setup using real INTC (Intel) data from LOBSTER, then extends it in two ways: conditioning on spread regimes, and replacing the Markovian order flow with a **Hawkes process** — but that decision didn't come out of nowhere, and there's a specific reason for it which we'll get to.

---

## Data

The project uses INTC LOBSTER data:
- `intc_messages.csv` — event-by-event record of what happened (order type, size, price, direction, time)
- `intc_orderbook.csv` — the state of the book at each event (ask/bid prices and volumes at multiple levels)

Order sizes are counted in units of 100 shares. Spread is measured in ticks.

---

## What was actually built

### Step 1 — Data Loading (`data_loader.py`)

Loads and aligns the message and orderbook files, detects the tick size, computes the spread in ticks, and splits the data into two **spread regimes**:

- **R1**: spread = 1 tick (tight spread)
- **R2**: spread ≥ 2 ticks (wider spread)

This regime split was an addition beyond the original paper. The paper mostly assumes one spread setting. Conditioning on spread turned out to matter quite a bit (more on that below).

---

### Step 2 — Markovian Calibration (`calibration.py`)

For each regime, this estimates the event rate matrices from the data:

- `alpha` — sell limit order arrival rate
- `beta` — buy limit order arrival rate
- `mu` — sell cancellation rate
- `gamma` — buy cancellation rate

Each rate depends on order size and relative price level. Calibration is basically: count how many events of each type happened in each regime, divide by how much time was spent in that regime. Saved to `calibrated_params.npz`.

---

### Step 3 — Markovian KBE Probability Surface (`kbe_engine.py`)

This is the core of the paper's contribution. Using the calibrated rates and the Kolmogorov backward equation, the engine computes:

```
P(ask price goes up | ask queue depth z1, bid queue depth z2, regime)
```

The state is simplified to just two numbers: `z1` (depth at best ask) and `z2` (depth at best bid). The output is a probability surface for R1 and R2, saved in `prob_lookup.npz`. It also computes `R2 - R1` to see how much the spread changes things.

---

### Step 4 — Validation (`validation.py`)

This is where we checked whether the probabilities actually make sense. Three things were compared:

1. **KBE probability** — what the equation gives us
2. **Monte Carlo probability** — simulated the same process many times and counted outcomes
3. **Empirical probability** — looked at what actually happened in the real INTC data

Results:

```
R1:
  KBE vs MC MAE        = 0.0682
  KBE vs Empirical MAE = 0.2381
  MC  vs Empirical MAE = 0.2104

R2:
  KBE vs MC MAE        = 0.1132
  KBE vs Empirical MAE = 0.3098
  MC  vs Empirical MAE = 0.2654
```

So KBE and Monte Carlo agree quite well, especially in R1. That means the math is internally consistent. The bigger gap is between both model outputs and the actual data — which is expected, since the Markovian assumption is a simplification of reality.

---

### Step 5 — Spread Analysis (`spread_analysis.py`)

Looked at how the probability surface shifts between R1 and R2:

```
Mean P(ask up) in R1 = 0.5039
Mean P(ask up) in R2 = 0.4476
Difference           = -0.0564
```

On average, the probability of an ask-price increase drops when the spread widens. About 61% of states showed a lower probability in R2. This makes intuitive sense — when the spread is wider, there is more room for a sell order to land inside the spread and push the ask price down.

---

### Step 6 — Why Hawkes? The data told us to move on

At this point the Markovian model was working. But there was something uncomfortable sitting in the background — the assumption that order arrivals behave like a Poisson process. Poisson means memoryless. Each order arrives independently, with no knowledge of what just happened before it.

So we actually went and checked this against the data.

For a Poisson process, the inter-arrival times should be roughly exponentially distributed. A clean way to test this is to look at the **Coefficient of Variation (CV)** of the inter-arrival times. CV is just the standard deviation divided by the mean:

```
CV = standard deviation of inter-arrival times / mean of inter-arrival times
```

For a perfect Poisson process, CV should be exactly **1.0** — because exponential distributions have equal mean and standard deviation by definition.

When we computed the CV on the actual INTC inter-arrival data, it came out to around **3.94**.

That's almost four times what Poisson would predict. This means the inter-arrival times are far more spread out than Poisson expects — there are long quiet periods and then sudden bursts of activity packed very close together. The data is heavily **clustered**, not random and memoryless. Poisson simply cannot explain a CV this high, so continuing with a Poisson order flow assumption would mean knowingly ignoring something the data is very clearly showing. That's when we decided to move to a Hawkes process.

---

### Step 7 — What is a Hawkes Process? (`hawkes_data_loader.py`)

A Hawkes process is a self-exciting point process. The core idea is that every time an event happens, it temporarily increases the rate of future events. The intensity (rate of events) at any time t is given by:

#### λ(t) = μ + Σ α · e^(−β(t−tᵢ))

Breaking this down in plain words:

- **μ (mu)** — the baseline rate. Even if nothing has happened recently, events still arrive at this background level.
- **α (alpha)** — excitation strength. Every past event adds a jump of size α to the current rate.
- **β (beta)** — decay rate. That jump doesn't last forever — it fades away exponentially at speed β.
- **The sum** — runs over all past event times tᵢ, so the more recent events there were, the higher the current rate is right now.

In simple terms: if a lot of orders just came in, the model expects more orders to keep coming in the near future. The excitement builds and then gradually dies down. This is much closer to how real order books behave — and much more consistent with a CV of 3.94 than a Poisson process ever could be.

One useful number to summarize the self-excitation is the **branching ratio** = α/β. If this is close to 0, the process behaves almost like Poisson. If it's close to 1, clustering is strong.

Inter-arrival streams are prepared for each combination of regime, event type, depth bin, and relative price level. Output saved to `interarrival_times.npz`.

---

### Step 8 — Hawkes Calibration (`hawkes_calibration.py`)

Fits the three Hawkes parameters — mu, alpha, beta — for each event-type / depth / excitation bin combination. Saved as `hawkes_params.npz`. Shape is `(4 event types, 10 depth bins, 6 excitation bins)` per regime.

---

### Step 9 — State Lift (`state_lift.py`)

Hawkes processes are not Markovian by nature — the current rate depends on the whole history of past events, not just the current state. To still use the KBE framework from the paper, the project introduces an approximate lift: an extra state variable `xi` (excitation bin) is added to the existing state `(z1, z2)`. This creates a larger Markov-like state space that approximates the memory of the Hawkes process. The resulting sparse generator matrices `L_H_R1.npz` and `L_H_R2.npz` have 2400 states each.

---

### Step 10 — Hawkes KBE (`hawkes_kbe_engine.py`)

Computes the Hawkes-based probability surface. One issue that came up during development: the cancellation rates were not monotone in queue depth — deeper queues were showing higher cancel rates, which doesn't make physical sense. This was fixed using isotonic regression on the depth axis for cancellation events. After fixing:

```
R1: z1 monotonicity = 100%, z2 monotonicity = 100%
R2: z1 monotonicity = 100%, z2 monotonicity = 100%
```

---

### Step 11 — Hawkes Validation (`hawkes_validation.py`)

Same three-way comparison as the Markovian model:

```
Hawkes R1:
  KBE vs MC MAE        = 0.0916
  KBE vs Empirical MAE = 0.3045
  MC  vs Empirical MAE = 0.3595
```

The Hawkes model's internal KBE-vs-MC agreement is decent. But the gap to empirical data is larger than the Markovian model. The report also shows:

```
Hawkes gap closure = -18.1%
```

This means the Hawkes Monte Carlo result actually moved a bit further from the empirical data compared to the KBE baseline — it didn't close the gap, it slightly widened it.

---

### Step 12 — Hawkes Diagnostics (`hawkes_analysis.py`)

Branching ratios (alpha/beta) per event type:

```
R1:
  sell_submit  = 0.051
  buy_submit   = 0.070
  sell_cancel  = 0.020
  buy_cancel   = 0.043

R2:
  sell_submit  = 0.00047
  buy_submit   = 0.00136
  sell_cancel  = 0.00061
  buy_cancel   = 0.00170
```

These are very small — especially in R2. A branching ratio near 0 means the Hawkes process is barely different from Poisson after calibration. This seems contradictory given the CV of 3.94, but it's almost certainly a data size problem. With only one trading day, the rarer R2 states don't have enough events for calibration to reliably detect the self-excitation signal even if it's genuinely there in the data.

---


## Which model is better?

Depends on what you're measuring:

| Metric | Better model |
|---|---|
| KBE vs MC internal agreement | Markovian |
| Empirical probability accuracy (MAE) | Markovian |
| AUC / forecast ranking | Hawkes |
| Net trading performance | Both negative, Hawkes slightly less bad |

The Markovian model is cleaner and more stable. It fits the paper's framework well, the KBE and MC agree nicely, and it's easier to interpret.

The Hawkes model is more ambitious and theoretically more justified — the CV of 3.94 makes a strong case that Poisson is the wrong assumption. But the fitted branching ratios are tiny, which means the Hawkes process is behaving almost like Poisson after calibration anyway. More data would likely change this picture significantly. It does show better AUC in the backtest, which suggests it's ranking states better, but not strongly enough yet to overcome transaction costs.

So to put it simply: **Markovian wins on validation metrics right now, but Hawkes is the right direction — the data supports it, the current dataset just isn't large enough to let it fully express itself.**

---

## Have we just calculated a probability, or also checked if it's correct?

Both. The project:

1. Computed `P(ask up)` from the KBE
2. Computed the same probability from Monte Carlo simulation
3. Estimated the same probability directly from real INTC data
4. Compared all three using MAE, RMSE, and correlation

So yes — the probabilities were calculated and then checked. The internal KBE-vs-MC agreement is pretty good. The model-vs-reality gap is larger, which is expected given the simplifications made.

---

## What's missing / future work

Right now the main limitation is **data size**. Quality LOBSTER data is not freely available, so the project currently runs on a single day of INTC sample data. With more data:

- The empirical probabilities would be much more stable
- Hawkes calibration would improve significantly — especially in R2 where the branching ratios are near zero right now, almost certainly because there aren't enough R2 events in one trading day to detect the excitation pattern reliably
- The CV of 3.94 suggests the clustering signal is real and strong in the raw data — better calibration with more data should let the Hawkes model actually express that and hopefully improve the empirical MAE
- The backtest could be made more realistic — larger held-out period, multiple days, proper transaction cost modeling, regime-conditional position sizing

The plan is to eventually run this on multiple days of data, properly backtest the regime-conditional probability surfaces, and see if the Hawkes model's AUC advantage translates into something tradeable once the calibration is more robust.

---

## File Summary

| File | What it does |
|---|---|
| `data_loader.py` | Load LOBSTER data, assign regimes |
| `calibration.py` | Calibrate Markovian rate matrices |
| `kbe_engine.py` | Solve KBE, produce probability surface |
| `validation.py` | KBE vs MC vs empirical comparison |
| `spread_analysis.py` | Analyze R1 vs R2 probability differences |
| `hawkes_data_loader.py` | Prepare inter-arrival streams for Hawkes |
| `hawkes_calibration.py` | Fit Hawkes mu/alpha/beta parameters |
| `state_lift.py` | Build lifted Markov state space with excitation bin |
| `hawkes_kbe_engine.py` | Compute Hawkes probability surface |
| `hawkes_validation.py` | Hawkes KBE vs MC vs empirical comparison |
| `hawkes_analysis.py` | Hawkes diagnostics and branching ratio analysis |
| `model_backtest.ipynb` | Trading backtest using both model probability surfaces |

---

*Based on: Cont, Degond, Xuan — "A mathematical framework for modelling order book dynamics" (2023)*
