"""
Parse the ``DESCRIPTION`` column of ``SPGE_MARKETDATA_SHARE.MDV2.PRICEDATA``
into structured, human-readable fields.

A lot of useful information in this dataset lives as **substrings inside
``DESCRIPTION``** rather than in dedicated columns. For example:

    "NYMEX WTI Calendar Swap Apr 2026 Elect"

contains the absolute maturity month (``Apr 2026``) and the settlement style
(``Elect`` = electronic), neither of which appear anywhere else in
``PRICEDATA`` (the SYMBOL ``XNCW001`` only carries the *relative* position,
not the absolute calendar month).

This module decodes those strings into clean fields:

>>> from src.parse_description import parse_description
>>> parse_description("NYMEX WTI Calendar Swap Apr 2026 Elect").to_dict()
{'family': 'nymex_wti_calendar_swap',
 'exchange': 'NYMEX',
 'commodity': 'WTI Crude Oil',
 'contract_type': 'Calendar Swap',
 'maturity_month': 'Apr',
 'maturity_year': 2026,
 'settlement_style': 'Electronic',
 ...}

It currently recognises the four families that account for the vast majority
of rows in this share:

  1. NYMEX WTI Calendar Swap   (XNCW***)
  2. NYMEX NY ULSD             (XNHO***)
  3. ICE Gas Oil               (XILO***)
  4. DTN retail rack           (DP*****)

Anything else falls back to ``family='other'`` with the raw description
preserved.

Run the module directly to see a worked-example table::

    python -m src.parse_description
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional

import pandas as pd


# ----------------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------------

# "NYMEX WTI Calendar Swap Apr 2026 Elect"
_RE_NYMEX_WTI_SWAP = re.compile(
    r"^NYMEX\s+WTI\s+Calendar\s+Swap\s+(?P<mo>\w{3})\s+(?P<yr>\d{4})\s+(?P<settle>\w+)$",
    re.IGNORECASE,
)

# "NYMEX NY ULSD 01-Mo Floor" or "NYMEX Globex NY ULSD 01-Mo Elect"
_RE_NYMEX_ULSD = re.compile(
    r"^NYMEX\s+(?:Globex\s+)?NY\s+ULSD\s+(?P<pos>\d{2})-Mo\s+(?P<settle>\w+)$",
    re.IGNORECASE,
)

# "ICE Gas Oil 01-Mo Comb"
_RE_ICE_GASOIL = re.compile(
    r"^ICE\s+Gas\s+Oil\s+(?P<pos>\d{2})-Mo\s+(?P<settle>\w+)$",
    re.IGNORECASE,
)

# "Intraday NYMEX RBOB Mo01"  /  "Intraday NYMEX NY ULSD Mo01"
_RE_INTRADAY = re.compile(
    r"^Intraday\s+NYMEX\s+(?P<commodity>RBOB|NY\s+ULSD|WTI|HO)\s+Mo(?P<pos>\d{2})$",
    re.IGNORECASE,
)

# DTN retail rack:
#   "DTN <Product> <City> <2-letter state> [<Brand>] <Suffix>"
# The 2-letter state code anchors the parse.
_RE_DTN_STATE = re.compile(r"\s([A-Z]{2})\s")

# DTN product prefixes (longest match wins). We keep this list short and
# extend as needed -- unrecognised prefixes fall back to a single-token guess.
_DTN_PRODUCTS = [
    "Unl Prem", "Unl Reg", "Unl Mid", "Mid Unl",
    "Premium Unleaded", "Regular Unleaded", "Mid-Grade Unleaded",
    "ULSD On-Hwy", "ULSD Off-Hwy",
    "ULSD Red", "ULSD Clear", "ULSD Dyed",
    "No.2 Diesel", "No.1 Diesel",
    "Premium Diesel", "Diesel Red", "Diesel Clear",
    "Heating Oil", "Kero",
    "Jet A", "Jet Fuel",
    "Diesel", "ULSD", "Unl",
]

# Settlement-style decoder
_SETTLE_MAP = {
    "elect": "Electronic",
    "floor": "Floor (open-outcry)",
    "comb":  "Combined (cleared + electronic)",
    "pit":   "Floor (pit)",
}

# Month abbreviation -> integer month number
_MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)}


# ----------------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------------

@dataclass
class ParsedDescription:
    """Structured view of a Platts/CME ``DESCRIPTION`` string.

    All fields are optional -- only the ones relevant to the matched family
    are populated. ``family`` is always set; for unrecognised inputs it is
    ``'other'`` and ``raw`` carries the original string.
    """

    family: str = "unknown"
    raw: str = ""

    # General fields
    exchange: Optional[str] = None
    commodity: Optional[str] = None
    region: Optional[str] = None
    contract_type: Optional[str] = None
    settlement_style: Optional[str] = None

    # Maturity (futures / swaps)
    maturity_month: Optional[str] = None       # 'Apr', 'Aug', ...
    maturity_month_num: Optional[int] = None   # 1..12
    maturity_year: Optional[int] = None        # 2026
    maturity_position: Optional[int] = None    # 1..69 (months ahead of asof)

    # DTN retail rack
    product: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    brand: Optional[str] = None
    branded: Optional[str] = None              # 'Branded' / 'Unbranded' / 'Aggregate ...'

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def parse_description(desc: str) -> ParsedDescription:
    """Parse a single ``DESCRIPTION`` string into a :class:`ParsedDescription`."""
    if not isinstance(desc, str) or not desc.strip():
        return ParsedDescription()

    s = desc.strip()

    # 1. NYMEX WTI Calendar Swap
    m = _RE_NYMEX_WTI_SWAP.match(s)
    if m:
        mo = m["mo"].title()
        return ParsedDescription(
            family="nymex_wti_calendar_swap",
            raw=s,
            exchange="NYMEX",
            commodity="WTI Crude Oil",
            contract_type="Calendar Swap",
            settlement_style=_SETTLE_MAP.get(m["settle"].lower(), m["settle"]),
            maturity_month=mo,
            maturity_month_num=_MONTH_NUM.get(mo),
            maturity_year=int(m["yr"]),
        )

    # 2. NYMEX NY ULSD
    m = _RE_NYMEX_ULSD.match(s)
    if m:
        return ParsedDescription(
            family="nymex_ny_ulsd",
            raw=s,
            exchange="NYMEX",
            commodity="Ultra Low Sulfur Diesel",
            region="New York Harbor",
            contract_type="Future",
            settlement_style=_SETTLE_MAP.get(m["settle"].lower(), m["settle"]),
            maturity_position=int(m["pos"]),
        )

    # 3. ICE Gas Oil
    m = _RE_ICE_GASOIL.match(s)
    if m:
        return ParsedDescription(
            family="ice_gas_oil",
            raw=s,
            exchange="ICE",
            commodity="Low-Sulfur Gas Oil",
            region="Northwest Europe (London)",
            contract_type="Future",
            settlement_style=_SETTLE_MAP.get(m["settle"].lower(), m["settle"]),
            maturity_position=int(m["pos"]),
        )

    # 4. Intraday NYMEX (RBOB / NY ULSD / WTI / HO)
    m = _RE_INTRADAY.match(s)
    if m:
        commodity_token = re.sub(r"\s+", " ", m["commodity"].strip().upper())
        commodity = {
            "RBOB":    "RBOB Gasoline",
            "NY ULSD": "Ultra Low Sulfur Diesel",
            "WTI":     "WTI Crude Oil",
            "HO":      "Heating Oil",
        }.get(commodity_token, commodity_token)
        return ParsedDescription(
            family="nymex_intraday",
            raw=s,
            exchange="NYMEX (intraday snap)",
            commodity=commodity,
            contract_type="Future (intraday)",
            settlement_style="Intraday quote",
            maturity_position=int(m["pos"]),
        )

    # 5. DTN retail rack
    if s.upper().startswith("DTN "):
        return _parse_dtn(s)

    # 6. Fallback
    return ParsedDescription(family="other", raw=s)


def parse_descriptions(descriptions: Iterable[str]) -> pd.DataFrame:
    """Parse a sequence of descriptions and return a tidy ``DataFrame``."""
    rows = [parse_description(d).to_dict() for d in descriptions]
    return pd.DataFrame(rows)


def add_parsed_columns(df: pd.DataFrame, *, source_col: str = "DESCRIPTION",
                       prefix: str = "") -> pd.DataFrame:
    """Return ``df`` with new structured columns parsed from ``source_col``.

    Existing columns are preserved; new ones are added with an optional
    ``prefix`` (e.g. ``prefix='desc_'`` -> ``desc_exchange``).
    """
    parsed = parse_descriptions(df[source_col])
    if prefix:
        parsed = parsed.add_prefix(prefix)
    return pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)


# ----------------------------------------------------------------------------
# DTN parser (internal)
# ----------------------------------------------------------------------------

def _parse_dtn(s: str) -> ParsedDescription:
    """Decode a DTN retail-rack description.

    Format: ``DTN <Product> <City> <ST> [<Brand>] <Suffix>``
    Suffix is one of: ``Br`` (branded), ``Unb`` (unbranded),
    ``BrAvg`` / ``BrMAvg`` / ``UnbAvg`` (aggregates).
    """
    body = s[4:].strip()  # drop "DTN "

    m = _RE_DTN_STATE.search(" " + body + " ")
    if not m:
        return ParsedDescription(family="dtn", raw=s, exchange="DTN (retail rack)")

    state = m.group(1)
    head, _sep, tail = body.partition(f" {state} ")

    # Match head against the known product prefix list (longest first).
    product, city = None, None
    for p in sorted(_DTN_PRODUCTS, key=len, reverse=True):
        if head.lower().startswith(p.lower()):
            product = p
            city = head[len(p):].strip() or None
            break
    if product is None:
        # Fallback: assume product is the first 1-2 tokens, city is the rest.
        toks = head.split()
        if len(toks) >= 2:
            product, city = toks[0], " ".join(toks[1:])
        else:
            product = head or None

    # Tail = "<brand-tokens...> <suffix>" or just "<suffix>".
    brand, branded = None, None
    tail_toks = tail.strip().split()
    if tail_toks:
        last = tail_toks[-1]
        if last in {"Br", "Unb"}:
            branded = "Branded" if last == "Br" else "Unbranded"
            brand = " ".join(tail_toks[:-1]) or None
        elif last in {"BrAvg", "BrMAvg", "UnbAvg", "Avg", "MAvg",
                      "RkAvg", "RkMAvg", "Rk"}:
            branded = f"Aggregate ({last})"
            brand = " ".join(tail_toks[:-1]) or None
        else:
            # Unrecognised suffix -- treat the whole tail as the brand.
            brand = tail.strip()

    return ParsedDescription(
        family="dtn",
        raw=s,
        exchange="DTN (retail rack)",
        commodity=product,
        product=product,
        city=city,
        state=state,
        brand=brand,
        branded=branded,
    )


# ----------------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        "NYMEX WTI Calendar Swap Apr 2026 Elect",
        "NYMEX WTI Calendar Swap Dec 2030 Elect",
        "NYMEX WTI Calendar Swap Aug 2031 Elect",
        "NYMEX NY ULSD 01-Mo Floor",
        "NYMEX NY ULSD 12-Mo Floor",
        "ICE Gas Oil 01-Mo Comb",
        "ICE Gas Oil 06-Mo Comb",
        "DTN Unl Prem Pittsburgh PA Marathon Br",
        "DTN Unl Prem Pittsburgh PA Marathon Unb",
        "DTN Unl Prem Pittsburgh PA Petro Prod Unb",
        "DTN Unl Prem Pittsburgh PA BrAvg",
        "DTN Unl Reg Pittsburgh PA Gulf-GIE Unb",
        "Brent Mo01 Spore MAvg",   # Platts-style; falls back to 'other'
    ]
    df = parse_descriptions(examples)
    df.insert(0, "DESCRIPTION", examples)

    # Drop noisy/empty columns for a cleaner display.
    cols_to_show = [
        "DESCRIPTION", "family", "exchange", "commodity",
        "maturity_position", "maturity_month", "maturity_year",
        "settlement_style", "city", "state", "brand", "branded",
    ]
    df = df[cols_to_show]

    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))
