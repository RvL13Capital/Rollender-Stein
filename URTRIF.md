# URTRIF v3.0 — Canonical Specification

This is the **foundational specification document** of this project. The
original PDF lives at [`docs/URTRIF_v3.0.pdf`](docs/URTRIF_v3.0.pdf); this
markdown file transcribes its content for version control, search, and
diffability, and adds the **relationship to AVE** — the implementation in
this repo.

> **URTRIF v3.0 — Unified Real Total Return Index Framework.**
> Transforms any raw price series into a unified time series that
> represents real performance in constant purchasing power, so historical
> and present prices of different assets become directly comparable.

The repo's existing artefacts (`src/rollender_stein/`, the four
numéraires, the dashboard) are the **multi-numéraire generalization** of
URTRIF — same mathematical core, broader purchasing-power vocabulary.

---

## 1. Goal

Nominal stock prices are **not comparable** across long time horizons or
across different assets. Inflation, FX, corporate actions, and data gaps
distort the price. URTRIF transforms each raw price series into a unified
time series whose values represent real performance in constant purchasing
power — so historical and present prices of different assets become
directly comparable.

## 2. Order of Operations (daily, immutable)

| # | Step | What it does |
|---|---|---|
| **1** | Corporate Actions | Splits + dividends, local & day-exact |
| **2** | FX Conversion | Into base currency, with exact cross term |
| **3** | Inflation | Anchored via CPI, repainting-free |

The order is load-bearing. Reversing FX and inflation, or applying splits
after FX, produces silently wrong numbers.

## 3. Mathematical Definition

### 3.1 Local total-return factor (robust against gaps)

$$
r^{\text{local}}_t \;=\; \frac{P_t + D_t}{P_{t-1} / S_t} \;-\; 1
$$

where $P_t$ is the local-currency closing price, $D_t$ is the dividend
paid on day $t$ (zero on most days), and $S_t$ is the split ratio
declared on day $t$ (1 if no split). Dividing the **previous** price by
$S_t$ — rather than retroactively scaling the entire price history — is
what makes URTRIF reproducible across re-ingests when future splits land.

### 3.2 FX return

$$
r^{\text{FX}}_t \;=\; \frac{F_t}{F_{t-1}} \;-\; 1
$$

where $F_t$ is the local-currency-to-base-currency exchange rate.

### 3.3 Logarithmic base return (numerically stable, exact cross term)

$$
\log r^{\text{base}}_t \;=\; \log(1 + r^{\text{local}}_t) + \log(1 + r^{\text{FX}}_t)
$$

Log addition is the key to numerical stability over 12,500+ trading days.
The cross term $r^{\text{local}}_t \cdot r^{\text{FX}}_t$ is implicit and
exact in the log domain.

### 3.4 Nominal base index (anchored at 100)

$$
I^{\text{base}}_t \;=\; 100 \cdot \exp\!\left(\sum_{s \le t} \log r^{\text{base}}_s\right)
$$

The `cumsum + exp` form avoids floating-point drift that `cumprod` would
accumulate. Over 25 years of daily data, that drift is real and visible.

### 3.5 Real index (auto-anchored, purchasing-power-adjusted)

$$
I^{\text{real}}_t \;=\; I^{\text{base}}_t \cdot \frac{C_{t_0}}{C_t}
$$

where $C_t$ is the CPI on day $t$ (forward-filled from the most recent
release with `release_date <= t`), and $C_{t_0}$ is automatically derived
from the first valid CPI value in the window — no manual anchoring.

## 4. Production-Grade Data Engineering

| Property | Implementation |
|---|---|
| **Gap Trap Fix** | `ffill()` on close and FX before any `shift()` — no returns lost over holidays, halts, or NaN gaps |
| **Self-Anchoring** | $C_{t_0}$ derived from first valid CPI value — no manual parameter mistakes |
| **Numerical Stability** | Log addition + cumsum + exp instead of cumprod — no floating-point drift over 12,500+ trading days |
| **No Look-Ahead Bias** | CPI joined via `merge_asof(direction='backward')` |

## 5. Limitations

- Taxes, transaction costs, and survivorship bias are **not included**;
  they must be modeled separately.
- For multi-asset comparisons: apply the function separately to each
  series (same CPI base, asset-specific FX).
- Result: each `I_real` column is directly plottable and comparable —
  nominal price differences are eliminated.

---

## 6. Relationship to the AVE in this repo

URTRIF v3.0 produces **one** real-total-return index per asset
($I^{\text{real}}_t$). The AVE (`src/rollender_stein/`) produces
**four** parallel indices per asset, one per numéraire
$X \in \{\text{Time}, \text{Energy}, \text{Liquidity}, \text{Gold}\}$.

### 6.1 The mathematical bridge

URTRIF's $I^{\text{real}}_t$ with deflator $C_t$ is algebraically
equivalent to AVE's $\text{Asset\_in\_C}_t$ with $N_C(t) =
C_t / C_{t_0} \cdot 100$:

$$
\underbrace{I^{\text{real}}_t}_{\text{URTRIF}}
= I^{\text{base}}_t \cdot \frac{C_{t_0}}{C_t}
= \frac{P^{\text{TR}}_t}{P_{t_0}} \cdot \frac{C_{t_0}}{C_t}
= \underbrace{\frac{P^{\text{TR}}_t}{P_{t_0} \cdot C_t/C_{t_0}}}_{\text{= AVE Asset\_in\_C}_t}
$$

So **AVE is URTRIF generalized to multi-numéraire**. Where URTRIF picks
CPI as its single deflator, AVE picks four orthogonal-ish deflators and
makes the **divergence** between them the unit of analysis.

### 6.2 The four AVE numéraires as URTRIF instances

Each AVE numéraire is what URTRIF would call "the deflator $C_t$" if the
question were narrowed to that dimension:

| AVE numéraire | URTRIF deflator semantic |
|---|---|
| `N_Time` (AHETPI) | "Real performance in hours-of-labor purchasing power" |
| `N_Energy` (Brent → MWh) | "Real performance in MWh-of-energy purchasing power" |
| `N_Liq` (G3 broad money) | "Real performance in share-of-broad-money purchasing power" |
| `N_Gold` (raw GC=F) | "Real performance in ounces-of-gold purchasing power" |

URTRIF asks: *what is this asset's real return, period?* AVE asks: *which
deflation dimension does this asset's apparent gain come from, and where
do they diverge?* The first is a portfolio-reporting question; the second
is a forensic-measurement question.

### 6.3 Why AVE is more than URTRIF + 3 extra deflators

Three concrete extensions beyond a literal "run URTRIF four times":

1. **Bitemporal store with `release_date >= reference_date` CHECK** —
   AVE separates the date a value *describes* from the date the public
   *learned* it. URTRIF's "no look-ahead bias" is an output property; AVE
   makes it a schema invariant (Patches 01 + 02).
2. **First-release vintage data via FRED ALFRED** — AVE uses
   `output_type=4` so the deflator at calendar time `t` is what was
   actually published on or before `t`, not what BLS retroactively
   revised. URTRIF's spec doesn't formalize this distinction.
3. **Per-finding decision log
   ([`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md))** — every methodological
   choice (PBOC excluded, Kalman demoted, perspective-commitment as §7)
   is durable and reasoned. URTRIF's "production-grade" framing covers
   engineering; AVE adds methodological accountability.

### 6.4 Where AVE owes URTRIF a debt — and where it could borrow back

URTRIF gets two things sharper than AVE currently does:

| URTRIF technique | AVE current state | Could AVE adopt? |
|---|---|---|
| **Day-of-split anchoring** ($P_{t-1} / S_t$ at split day, leave history raw) | Uses yfinance `adj_close` retroactive split-adjustment ([KNOWN_LIMITATIONS L-6](KNOWN_LIMITATIONS.md)) | Yes — `marketcap.py::_cumulative_future_split_factor` already has the right primitive; could be ported into per-share valuation |
| **Explicit FX as a separate pipeline step** | FX implicit inside `N_Liq` only; non-USD-listed assets have no clean path | Yes — would require either a fifth numéraire `N_FX` or an explicit FX stage before the division |
| **Log-addition + cumsum for numerical stability** | AVE doesn't compound returns (uses pointwise division at each `t`) — moot | n/a |

These are queued as candidate improvements, not committed plans. See
[`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) L-6 (current) and any
future entries for the day-of-split / FX-axis discussion.

### 6.5 Where URTRIF is silent and AVE supplies the answer

Conversely, the URTRIF spec leaves several questions unspecified that
AVE has had to answer in implementation:

| Question | URTRIF spec | AVE answer |
|---|---|---|
| Is the CPI release-dated or reference-dated? | Silent | `release_date` strictly enforced (Patch 01 schema CHECK) |
| What about CPI's own annual revisions? | Silent | First-release via ALFRED `output_type=4` |
| What if the deflator stops being published (e.g. EZ M3 levels in 2023-11)? | Silent | Splice via growth-rate compounding (`extend_levels_with_growth`) |
| What if the deflator's anchor is missing (e.g. GC=F starts 2000-08-30, T0 is 2000-01-03)? | Silent | Anchor-fallback to first valid date with `RuntimeWarning` ([KNOWN_LIMITATIONS L-2](KNOWN_LIMITATIONS.md)) |
| Which deflator? | CPI | Four, with documented rationale per choice ([§7 Perspective Commitment](AUDIT_DECISIONS.md)) |

URTRIF is a clean per-asset-real-TR-index recipe; AVE is what it takes
to actually run that recipe across decades, currencies, and methodological
revisions, four deflation dimensions in parallel.

---

## 7. Where this fits in the doc tree

This file is the **canonical specification**. Everything else is
implementation-and-discipline:

- [`README.md`](README.md) — quickstart and forensic principles
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline layers; **§7 (this
  file's section 6) explains how the architecture realizes URTRIF**
- [`INVARIANTS.md`](INVARIANTS.md) — what the implementation guarantees
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) — what it deliberately
  doesn't, including where it diverges from URTRIF (L-6 day-of-split)
- [`CHANGELOG.md`](CHANGELOG.md) — patch and feature history
- [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) — per-finding decision log,
  including §7 perspective commitment that selects the four AVE
  numéraires from the broader URTRIF-compatible deflator space
- [`CLAUDE.md`](CLAUDE.md) — methodological depth and runbooks

**Read order for someone new:** this file → README → ARCHITECTURE →
INVARIANTS → KNOWN_LIMITATIONS → CHANGELOG → AUDIT_DECISIONS → CLAUDE.

---

*The original specification PDF is preserved unchanged at
[`docs/URTRIF_v3.0.pdf`](docs/URTRIF_v3.0.pdf). When the PDF is updated to
v3.1 or later, this markdown should be updated in the same commit.*
