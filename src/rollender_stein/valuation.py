"""Phase 5 — the Division Array.

Given a target asset's daily nominal USD price and the available numéraires,
compute the four absolute-valuation arrays:

    Asset_in_X(t) = ( nominal_USD(t) / N_X(t) ) * 100
        for X in {Time, Liquidity, Gold, Energy}

The numéraires are already T0-anchored (N_X(T0) = 100 by construction), so the
result has units of **T0-deflated USD**: the asset's value expressed in
purchasing power equivalent to T0. At T0 itself, Asset_in_X(T0) equals the
asset's nominal T0 price — NOT 100. This is correct: BTC at $457 in 2014 and
SPX at $2,002 in 2000 should NOT both enter the phase space at [100, 100, 100],
because their real values differ by 4x.

The earlier per-asset "index to T0=100 first" step was wrong — it forced every
asset's trajectory to share the same synthetic origin regardless of nominal
scale, breaking cross-asset comparison.

``asset_indexed`` is still computed for transparency (asset relative to its T0
or first-valid value, normalized to 100) but is NOT used in the division.

Forensic rule (spec): for equity targets the input must be a TOTAL RETURN
series (e.g. ``^SP500TR``); price-only indexes underweight the wealth
generation curve when compared against yieldless numéraires like gold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rollender_stein.calendar import T0_DATE

NUMERAIRE_NAMES = ("Time", "Liquidity", "Gold", "Energy")


@dataclass(frozen=True)
class DivisionArray:
    nominal_usd: pd.Series
    asset_indexed: pd.Series
    asset_in_time: pd.Series | None
    asset_in_liquidity: pd.Series | None
    asset_in_gold: pd.Series | None
    asset_in_energy: pd.Series | None

    def to_frame(self) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {
            "nominal_usd": self.nominal_usd,
            "asset_indexed": self.asset_indexed,
        }
        if self.asset_in_time is not None:
            cols["asset_in_time"] = self.asset_in_time
        if self.asset_in_liquidity is not None:
            cols["asset_in_liquidity"] = self.asset_in_liquidity
        if self.asset_in_gold is not None:
            cols["asset_in_gold"] = self.asset_in_gold
        if self.asset_in_energy is not None:
            cols["asset_in_energy"] = self.asset_in_energy
        return pd.DataFrame(cols)


def build_division_array(
    nominal_asset_usd: pd.Series,
    *,
    n_time: pd.Series | None = None,
    n_liquidity: pd.Series | None = None,
    n_gold: pd.Series | None = None,
    n_energy: pd.Series | None = None,
    t0_date: pd.Timestamp = T0_DATE,
) -> DivisionArray:
    """Compute the four-dimensional division array.

    ``nominal_asset_usd`` is the raw USD price of the target. The division
    uses the nominal price directly (NOT pre-indexed to 100), so different
    assets enter the phase space at their real T0-deflated USD position
    rather than a synthetic per-asset origin.

    Outputs share the same index — the numéraires' master calendar if any are
    provided, otherwise the asset's own index — with the asset forward-filled
    to that calendar.

    ``asset_indexed`` is still computed (asset normalized to 100 at T0 if
    available, else first-valid date) but ONLY for inspection / transparency.
    It is not used in any ``asset_in_X`` calculation.
    """
    # asset_indexed: still anchor at T0 if available, else first-valid. This is
    # informational only — surface for tests/debugging, not used in division.
    if t0_date in nominal_asset_usd.index:
        anchor_for_indexed = float(nominal_asset_usd.loc[t0_date])
    else:
        loc = nominal_asset_usd.index.get_indexer(
            pd.DatetimeIndex([t0_date]), method="ffill"
        )[0]
        if loc >= 0:
            anchor_for_indexed = float(nominal_asset_usd.iloc[loc])
        else:
            first_valid = nominal_asset_usd.first_valid_index()
            if first_valid is None:
                raise RuntimeError("asset series is entirely empty/NaN")
            anchor_for_indexed = float(nominal_asset_usd.loc[first_valid])
    if not pd.notna(anchor_for_indexed) or anchor_for_indexed == 0:
        raise RuntimeError(f"asset anchor value is {anchor_for_indexed!r}; cannot index")

    # Settle on a common calendar — prefer the master calendar from the
    # numéraires; otherwise use the asset's own index.
    base_idx = None
    for n in (n_time, n_liquidity, n_gold, n_energy):
        if n is not None:
            base_idx = n.index
            break
    if base_idx is None:
        base_idx = nominal_asset_usd.index

    nominal_aligned = nominal_asset_usd.reindex(base_idx, method="ffill")
    indexed = (nominal_aligned / anchor_for_indexed) * 100.0

    def _ratio(num: pd.Series | None) -> pd.Series | None:
        if num is None:
            return None
        # Asset_in_X = (nominal_USD / N_X) * 100 — units are T0-deflated USD.
        ratio: pd.Series = (nominal_aligned / num.reindex(base_idx)) * 100.0
        out_name = str(num.name) if num.name is not None else "asset_in_X"
        return ratio.rename(out_name)

    return DivisionArray(
        nominal_usd=nominal_aligned.rename("nominal_usd"),
        asset_indexed=indexed.rename("asset_indexed"),
        asset_in_time=_ratio(n_time),
        asset_in_liquidity=_ratio(n_liquidity),
        asset_in_gold=_ratio(n_gold),
        asset_in_energy=_ratio(n_energy),
    )
