"""Phase 5 — the Division Array.

Given a target asset's daily nominal USD price and the available numéraires,
compute the four absolute-valuation arrays:

    Asset_indexed(t)        = (Asset_USD(t) / Asset_USD(T0)) * 100
    Asset_in_X(t)           = (Asset_indexed(t) / N_X(t)) * 100   for X in {Time, Liquidity, Gold, Energy}

By construction Asset_in_X(T0) == 100.0 whenever both the asset and N_X have
values at T0. Where N_X is NaN (e.g. N_Gold before 2006-01-03 in our setup),
the corresponding Asset_in_X is also NaN — the dashboard simply has no Z
coordinate for those days.

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

    ``nominal_asset_usd`` is the raw USD price of the target (NOT pre-indexed).
    The function indexes it to 100 at ``t0_date`` and produces:

      - ``asset_indexed`` : the T0=100 series
      - one ``asset_in_X`` series per provided numéraire

    All outputs share the same index — the union of asset and numéraire dates,
    aligned via forward-fill so every numéraire's calendar is honored.
    """
    if t0_date not in nominal_asset_usd.index:
        # Allow asset to have a slightly different calendar — find the latest
        # value at or before T0.
        loc = nominal_asset_usd.index.get_indexer([t0_date], method="ffill")[0]
        if loc < 0:
            raise RuntimeError(
                f"asset has no value at or before T0 ({t0_date.date()}); "
                f"earliest is {nominal_asset_usd.index[0].date()}",
            )
        anchor_value = float(nominal_asset_usd.iloc[loc])
    else:
        anchor_value = float(nominal_asset_usd.loc[t0_date])

    if not pd.notna(anchor_value) or anchor_value == 0:
        raise RuntimeError(f"asset value at T0 is {anchor_value!r}; cannot anchor")

    # Settle on a common calendar — prefer the master calendar implicit in the
    # numéraires; otherwise use the asset's own index.
    base_idx = None
    for n in (n_time, n_liquidity, n_gold, n_energy):
        if n is not None:
            base_idx = n.index
            break
    if base_idx is None:
        base_idx = nominal_asset_usd.index

    nominal_aligned = nominal_asset_usd.reindex(base_idx, method="ffill")
    indexed = (nominal_aligned / anchor_value) * 100.0

    def _ratio(num: pd.Series | None) -> pd.Series | None:
        if num is None:
            return None
        return ((indexed / num.reindex(base_idx)) * 100.0).rename(num.name or "asset_in_X")

    return DivisionArray(
        nominal_usd=nominal_aligned.rename("nominal_usd"),
        asset_indexed=indexed.rename("asset_indexed"),
        asset_in_time=_ratio(n_time),
        asset_in_liquidity=_ratio(n_liquidity),
        asset_in_gold=_ratio(n_gold),
        asset_in_energy=_ratio(n_energy),
    )
