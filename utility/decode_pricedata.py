"""
Decode PRICEDATA into local CSV files with human-readable metadata.

Two artefact types are produced:

1. Master symbol dimension (``data/decoded/symbols.csv``)
   One row per symbol, with raw + decoded metadata pulled from
   `REF_DATA_MARKETDATA` and overlaid with cleaner values from
   `PLATTS_SYMBOL_DIRECTORY` where available. Adds derived columns:
       - SYMBOL_FAMILY          (XNCW, XNHO, XILO, CFTC, EIA, DTN, OTHER)
       - CONTRACT_MONTH_NAME    (Jan-Dec, parsed from XNCW letter suffix)
       - CONTRACT_YEAR          (parsed from XNCW letter suffix)
       - MDC_CATEGORY           (Futures / Statistics / Racks / etc.)
       - PADD_REGION            (extracted from MDC_DESCRIPTION when relevant)

2. Per-family long-format price CSVs (``data/decoded/<family>.csv``)
   One row per (SYMBOL, ASSESSDATE, BATE) with VALUE plus BATE_NAME
   (Open/High/Low/Close/Open Interest/Volume/Unit Value).

Run:
    python -m src.decode_pricedata                       # safe default groups
    python -m src.decode_pricedata --group wti_swap      # one family
    python -m src.decode_pricedata --all                 # everything (large!)
    python -m src.decode_pricedata --since 2015-01-01    # date filter
    python -m src.decode_pricedata --gzip                # compress output
    python -m src.decode_pricedata --force               # rewrite existing
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from .snowflake_client import SnowflakeClient


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PRC = '"SPGE_MARKETDATA_SHARE"."MDV2"."PRICEDATA"'
RDM = '"SPGE_MARKETDATA_SHARE"."MDV2"."REF_DATA_MARKETDATA"'
PSD = '"HOUSTONPRODUCTSDESK"."BRONZE"."PLATTS_SYMBOL_DIRECTORY"'

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "data" / "verified"
OUT = ROOT / "data" / "decoded"
OUT.mkdir(parents=True, exist_ok=True)


# Verified empirically by `verify_mappings.py`.
BATE_LEGEND = {
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "e": "Open Interest",
    "w": "Volume",
    "u": "Unit Value",
}


# NYMEX month-letter codes (industry convention)
MONTH_LETTERS = {
    "F": ("Jan", 1), "G": ("Feb", 2), "H": ("Mar", 3),
    "J": ("Apr", 4), "K": ("May", 5), "M": ("Jun", 6),
    "N": ("Jul", 7), "Q": ("Aug", 8), "U": ("Sep", 9),
    "V": ("Oct", 10), "X": ("Nov", 11), "Z": ("Dec", 12),
}


# --------------------------------------------------------------------------
# Family definitions
# --------------------------------------------------------------------------
@dataclass
class Group:
    name: str
    description: str
    where_clause: str
    skip_by_default: bool = False
    expected_rows: str = ""


GROUPS: dict[str, Group] = {
    "wti_swap": Group(
        name="wti_swap",
        description="NYMEX WTI calendar swaps (XNCW%)",
        where_clause="SYMBOL LIKE 'XNCW%'",
        expected_rows="~3.7M",
    ),
    "nymex_ulsd_futures": Group(
        name="nymex_ulsd_futures",
        description="NYMEX Heating Oil / ULSD futures (XNHO%)",
        where_clause="SYMBOL LIKE 'XNHO%'",
        expected_rows="~2.3M",
    ),
    "ice_gasoil_futures": Group(
        name="ice_gasoil_futures",
        description="ICE Gas Oil futures (XILO%)",
        where_clause="SYMBOL LIKE 'XILO%'",
        expected_rows="~3.4M",
    ),
    "nymex_rbob_intraday": Group(
        name="nymex_rbob_intraday",
        description="NYMEX RBOB Gasoline intraday (XUHU%)",
        where_clause="SYMBOL LIKE 'XUHU%'",
        expected_rows="~46k",
    ),
    "cftc_positioning": Group(
        name="cftc_positioning",
        description="CFTC Commitments-of-Traders + Bank Position Reports (MDC=CM)",
        where_clause="MDC = 'CM'",
        expected_rows="~1.8M",
    ),
    "eia_inventory": Group(
        name="eia_inventory",
        description="EIA WPSR inventory & supply stats (PADD-level)",
        where_clause="MDC IN ('AS','AO','AT','AH','AF','AI')",
        expected_rows="~950k",
    ),
    "spot_oil_products": Group(
        name="spot_oil_products",
        description="Spot/physical product markets (NY Harbor, USGC, Singapore, etc.)",
        where_clause=("MDC IN ('EB','UG','AG','CS','PN','PL','CJ','CX',"
                      "'UW','UZ','WC','PG','PZ')"),
        skip_by_default=True,
        expected_rows="~10M",
    ),
    "bunkers": Group(
        name="bunkers",
        description="Marine fuel (bunker) prices — global, US, Latin America",
        where_clause="MDC IN ('BWD','BA','BL')",
        skip_by_default=True,
        expected_rows="~5.1M",
    ),
    "fx": Group(
        name="fx",
        description="Foreign exchange rates (3rd-party + Platts MOC)",
        where_clause="MDC = 'FX'",
        expected_rows="~470k",
    ),
    "dtn_retail_rack": Group(
        name="dtn_retail_rack",
        description="DTN US/Canada retail-rack truck-loading prices",
        where_clause="DESCRIPTION LIKE 'DTN %'",
        skip_by_default=True,
        expected_rows="~30M+ (large!)",
    ),
}


# Default groups when nothing is specified.
DEFAULT_GROUPS = [
    "wti_swap",
    "nymex_ulsd_futures",
    "ice_gasoil_futures",
    "nymex_rbob_intraday",
    "cftc_positioning",
    "eia_inventory",
    "fx",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _ext(gzip: bool) -> str:
    return ".csv.gz" if gzip else ".csv"


def _save_csv(df: pd.DataFrame, path: Path, gzip: bool) -> None:
    if gzip and path.suffix != ".gz":
        path = path.with_suffix(path.suffix + ".gz")
    df.to_csv(path, index=False, compression="gzip" if gzip else None)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  -> wrote {path.name}  ({len(df):,} rows, {size_mb:.1f} MB)")


PADD_MDCS = {"NE", "NY", "AM", "SA", "NA", "MW", "NI", "WO", "SM",
             "GC", "AA", "NR", "NB", "SB", "PR", "JPR"}
EIA_MDCS = {"AS", "AO", "AT", "AH", "AF", "AI"}
BUNKER_MDCS = {"BWD", "BA", "BL"}


def derive_symbol_family(symbol: str, mdc: str | None, description: str | None) -> str:
    """Classify a symbol into a family using SYMBOL pattern + MDC + DESCRIPTION.

    More reliable than symbol-prefix alone -- SYMBOL prefixes overlap (e.g.
    'DP' covers both DTN retail rack AND Platts Dead Prompt assessments).
    """
    if not isinstance(symbol, str):
        return "OTHER"

    # Futures/swap families have unique SYMBOL prefixes
    if symbol.startswith("XNCW"):
        return "WTI_SWAP"
    if symbol.startswith("XNHO"):
        return "NYMEX_ULSD"
    if symbol.startswith("XILO"):
        return "ICE_GASOIL"
    if symbol.startswith("XUHU"):
        return "NYMEX_RBOB"

    # MDC-driven classifications (more reliable than symbol prefix)
    mdc_tokens = set()
    if isinstance(mdc, str):
        mdc_tokens = {t.strip() for t in mdc.split(",")}

    if "CM" in mdc_tokens:
        return "CFTC"
    if mdc_tokens & EIA_MDCS:
        return "EIA"
    if mdc_tokens & BUNKER_MDCS:
        return "BUNKERS"
    if "FX" in mdc_tokens:
        return "FX"

    # Retail rack: PADD-coded MDC + description begins with "DTN "
    if mdc_tokens & PADD_MDCS:
        if isinstance(description, str) and description.startswith("DTN "):
            return "RETAIL_RACK_DTN"
        return "RETAIL_RACK_OTHER"

    return "OTHER"


def parse_xncw_calendar(symbol: str) -> tuple[str | None, int | None]:
    """For symbols like XNCWF19 (Jan 2019), return ('Jan', 2019).

    Returns (None, None) for plain XNCW001 or non-XNCW symbols.
    """
    if not isinstance(symbol, str) or len(symbol) != 7:
        return (None, None)
    if not symbol.startswith("XNCW"):
        return (None, None)
    suffix = symbol[4:]
    if not suffix or not suffix[0].isalpha():
        return (None, None)
    letter = suffix[0]
    yr2 = suffix[1:3]
    if letter not in MONTH_LETTERS or not yr2.isdigit():
        return (None, None)
    name, _ = MONTH_LETTERS[letter]
    year = 2000 + int(yr2)
    return (name, year)


# --------------------------------------------------------------------------
# Master symbol dimension
# --------------------------------------------------------------------------
def decode_padd_region(mdc_desc: str) -> str | None:
    """Extract 'PADD 1 - New York Area' style region name."""
    if not isinstance(mdc_desc, str):
        return None
    # mdc_desc example: 'NY-Racks: PADD 1 - New York Area | AM-Racks: PADD 1 - Mid-Atlantic Area'
    # we just capture the first 'PADD N - <name>' substring
    import re
    m = re.search(r"PADD\s*(\d)\s*-\s*([^|]+?)(?:\s*\||$)", mdc_desc)
    if m:
        return f"PADD {m.group(1)} - {m.group(2).strip()}"
    return None


def decode_mdc_category(mdc_desc: str) -> str:
    """High-level grouping based on MDC_DESCRIPTION text."""
    if not isinstance(mdc_desc, str) or not mdc_desc:
        return "Other"
    text = mdc_desc.lower()
    if "futures" in text or "nearbys" in text or "swap" in text:
        return "Futures/Swaps"
    if "racks" in text or "padd" in text:
        return "Retail Racks"
    if "statistics" in text or "stats" in text:
        return "Statistics/Reports"
    if "bunkers" in text:
        return "Bunkers"
    if "lpg" in text:
        return "LPG"
    if "metals" in text:
        return "Metals"
    if "shipping" in text or "freight" in text:
        return "Shipping/Freight"
    if "products" in text:
        return "Spot Oil Products"
    if "foreign exchange" in text:
        return "FX"
    if "petchem" in text or "chemicals" in text:
        return "Petchem/Chemicals"
    if "carbon" in text:
        return "Carbon/ESG"
    return "Other"


def build_symbols_table(sf: SnowflakeClient) -> pd.DataFrame:
    """Pull RDM + PSD overlay into a master symbols dimension."""
    print("\n=== Building master symbols.csv ===")
    print("  Pulling REF_DATA_MARKETDATA (full catalog) ...")
    rdm = sf.read_sql(f"""
        SELECT
            SYMBOL,
            DESCRIPTION,
            COMMODITY,
            COMMODITY_GRADE,
            COMMODITY_FORM,
            EXCHANGE,
            CONTRACT_TYPE,
            SETTLEMENT_TYPE,
            BENCHMARK,
            DERIVATIVE_POSITION,
            DERIVATIVE_MATURITY_FREQUENCY,
            DELIVERY_REGION,
            DELIVERY_METHOD,
            DELIVERY_LOAD,
            UOM,
            CURRENCY,
            DECIMAL_PLACES,
            ACTIVE,
            API_GRAVITY,
            SULFUR,
            DENSITY,
            ASSESSMENT_FREQUENCY,
            DAY_OF_PUBLICATION,
            QUOTATION_STYLE,
            MDC                AS MDC_RAW_RDM,
            MDC_DESCRIPTION    AS MDC_DESC_RDM
        FROM   {RDM}
    """)
    print(f"     {len(rdm):,} rows from RDM")

    print("  Pulling PLATTS_SYMBOL_DIRECTORY (cleaner overlay) ...")
    psd = sf.read_sql(f"""
        WITH unnested AS (
          SELECT  psd.SYMBOL,
                  ARRAY_TO_STRING(PARSE_JSON(psd.MDC), ',')              AS MDC_PSD,
                  ARRAY_TO_STRING(PARSE_JSON(psd.MDC_DESCRIPTION), ' | ') AS MDC_DESC_PSD,
                  psd.EXCHANGE        AS EXCHANGE_PSD,
                  psd.BENCHMARK       AS BENCHMARK_PSD,
                  psd.COMMODITY       AS COMMODITY_PSD,
                  psd.COMMODITY_GRADE AS COMMODITY_GRADE_PSD,
                  psd.CONTRACT_TYPE   AS CONTRACT_TYPE_PSD,
                  psd.SETTLEMENT_TYPE AS SETTLEMENT_TYPE_PSD
          FROM   {PSD} psd
        )
        SELECT * FROM unnested
    """)
    print(f"     {len(psd):,} rows from PSD")

    df = rdm.merge(psd, on="SYMBOL", how="left")

    # COALESCE: PSD wins where present, else RDM
    def _coalesce(left: str, right: str) -> pd.Series:
        return df[left].where(df[left].notna() & (df[left] != ""), df[right])

    df["EXCHANGE"] = _coalesce("EXCHANGE_PSD", "EXCHANGE")
    df["BENCHMARK"] = _coalesce("BENCHMARK_PSD", "BENCHMARK")
    df["COMMODITY"] = _coalesce("COMMODITY_PSD", "COMMODITY")
    df["COMMODITY_GRADE"] = _coalesce("COMMODITY_GRADE_PSD", "COMMODITY_GRADE")
    df["CONTRACT_TYPE"] = _coalesce("CONTRACT_TYPE_PSD", "CONTRACT_TYPE")
    df["SETTLEMENT_TYPE"] = _coalesce("SETTLEMENT_TYPE_PSD", "SETTLEMENT_TYPE")
    df["MDC"] = _coalesce("MDC_PSD", "MDC_RAW_RDM")
    df["MDC_DESCRIPTION"] = _coalesce("MDC_DESC_PSD", "MDC_DESC_RDM")

    # Drop the helper columns
    df = df.drop(columns=[c for c in df.columns if c.endswith("_PSD")])
    df = df.drop(columns=["MDC_RAW_RDM", "MDC_DESC_RDM"])

    print("  Deriving symbol family / contract month / PADD / MDC category ...")
    df["SYMBOL_FAMILY"] = df.apply(
        lambda r: derive_symbol_family(r["SYMBOL"], r.get("MDC"), r.get("DESCRIPTION")),
        axis=1,
    )

    # Calendar month parse (only XNCW with letter suffix)
    parsed = df["SYMBOL"].map(parse_xncw_calendar)
    df["CONTRACT_MONTH_NAME"] = parsed.map(lambda x: x[0])
    df["CONTRACT_YEAR"] = parsed.map(lambda x: x[1])

    df["PADD_REGION"] = df["MDC_DESCRIPTION"].map(decode_padd_region)
    df["MDC_CATEGORY"] = df["MDC_DESCRIPTION"].map(decode_mdc_category)

    # Final column order
    cols = [
        # identity
        "SYMBOL", "DESCRIPTION", "SYMBOL_FAMILY",
        # commodity / grade
        "COMMODITY", "COMMODITY_GRADE", "COMMODITY_FORM",
        # exchange / contract
        "EXCHANGE", "CONTRACT_TYPE", "SETTLEMENT_TYPE", "BENCHMARK",
        "DERIVATIVE_POSITION", "DERIVATIVE_MATURITY_FREQUENCY",
        "CONTRACT_MONTH_NAME", "CONTRACT_YEAR",
        # MDC
        "MDC", "MDC_DESCRIPTION", "MDC_CATEGORY", "PADD_REGION",
        # delivery
        "DELIVERY_REGION", "DELIVERY_METHOD", "DELIVERY_LOAD",
        # pricing units
        "UOM", "CURRENCY", "DECIMAL_PLACES",
        # chemistry
        "API_GRAVITY", "SULFUR", "DENSITY",
        # publication
        "ASSESSMENT_FREQUENCY", "DAY_OF_PUBLICATION", "QUOTATION_STYLE",
        # admin
        "ACTIVE",
    ]
    return df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------------
# Per-family decoded long-format CSV
# --------------------------------------------------------------------------
def decode_group(
    sf: SnowflakeClient,
    group: Group,
    *,
    since: str | None,
    gzip: bool,
    force: bool,
) -> bool:
    """Pull one family's price rows and save as long-format CSV.
    Returns True iff written, False if skipped (e.g. file exists).
    """
    out_path = OUT / f"{group.name}{_ext(gzip)}"
    print(f"\n=== {group.name}  --  {group.description}  (expected {group.expected_rows}) ===")

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  -> file exists ({out_path.name}, {size_mb:.1f} MB). Skipping (--force to overwrite).")
        return False

    where = group.where_clause
    if since:
        where = f"({where}) AND ASSESSDATE >= '{since}'"

    # Quick row count first (so the user knows what's coming)
    cnt_sql = f"SELECT COUNT(*) AS N FROM {PRC} WHERE {where}"
    n = int(sf.read_sql(cnt_sql).iloc[0, 0])
    print(f"  Estimated rows: {n:,}")

    if n == 0:
        print("  (no rows; skipping)")
        return False

    t0 = time.time()
    print("  Pulling ...")
    # Keep ASSESSDATE as TIMESTAMP -- some families (XUHU, DTN) carry true
    # intraday timestamps; daily families just have midnight.
    df = sf.read_sql(f"""
        SELECT
            ASSESSDATE,
            SYMBOL,
            BATE,
            VALUE,
            UOM,
            CURRENCY,
            ISCORRECTED,
            MDC,
            DESCRIPTION
        FROM   {PRC}
        WHERE  {where}
        ORDER  BY SYMBOL, ASSESSDATE, BATE
    """)
    elapsed = time.time() - t0
    print(f"  Pulled {len(df):,} rows in {elapsed:.1f}s")

    df["BATE_NAME"] = df["BATE"].map(BATE_LEGEND).fillna("Unknown")

    # Reorder for readability
    df = df[[
        "ASSESSDATE", "SYMBOL", "BATE", "BATE_NAME", "VALUE",
        "UOM", "CURRENCY", "ISCORRECTED", "MDC", "DESCRIPTION",
    ]]

    _save_csv(df, out_path, gzip)
    return True


# --------------------------------------------------------------------------
# Legend / dictionary CSVs (small, copied/refreshed)
# --------------------------------------------------------------------------
def write_legends(gzip: bool) -> None:
    """Copy verified legends and write a fresh BATE legend."""
    print("\n=== Writing legends ===")

    bate_path = OUT / f"bate_legend{_ext(gzip)}"
    bate_df = pd.DataFrame(
        [{"BATE": k, "BATE_NAME": v} for k, v in BATE_LEGEND.items()]
    )
    _save_csv(bate_df, bate_path, gzip)

    # Copy mdc_codes from verified/
    src = VERIFIED / "mdc_codes.csv"
    if src.exists():
        dst = OUT / f"mdc_codes{_ext(gzip)}"
        if gzip:
            pd.read_csv(src).to_csv(dst, index=False, compression="gzip")
        else:
            shutil.copy2(src, dst)
        print(f"  -> copied {src.name} -> {dst.name}")

    # NYMEX month-letter codes
    months_path = OUT / f"nymex_month_codes{_ext(gzip)}"
    months_df = pd.DataFrame(
        [{"LETTER": k, "MONTH": v[0], "MONTH_NUM": v[1]}
         for k, v in MONTH_LETTERS.items()]
    )
    _save_csv(months_df, months_path, gzip)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode PRICEDATA into local CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available groups:\n"
            + "\n".join(
                f"  {g.name:<22} {g.description}"
                f"   ({g.expected_rows}{' [opt-in]' if g.skip_by_default else ''})"
                for g in GROUPS.values()
            )
        ),
    )
    parser.add_argument("--group", action="append",
                        help="Run a specific group (repeatable). "
                             "If omitted, runs the safe defaults.")
    parser.add_argument("--all", action="store_true",
                        help="Run every group, including the large opt-in ones.")
    parser.add_argument("--since", default=None,
                        help="Only ASSESSDATE >= this (e.g. 2015-01-01). "
                             "Default: pull full history.")
    parser.add_argument("--gzip", action="store_true",
                        help="Save CSVs gzip-compressed (.csv.gz).")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite output files even if they already exist.")
    parser.add_argument("--no-symbols", action="store_true",
                        help="Skip rebuilding the master symbols.csv.")
    args = parser.parse_args(argv)

    if args.all:
        groups = list(GROUPS.values())
    elif args.group:
        unknown = [g for g in args.group if g not in GROUPS]
        if unknown:
            print(f"Unknown group(s): {unknown}", file=sys.stderr)
            print(f"Available: {list(GROUPS.keys())}", file=sys.stderr)
            return 2
        groups = [GROUPS[g] for g in args.group]
    else:
        groups = [GROUPS[name] for name in DEFAULT_GROUPS]

    print(f"Output directory: {OUT}")
    print(f"Groups to process: {[g.name for g in groups]}")
    if args.since:
        print(f"Date filter: ASSESSDATE >= {args.since}")

    t_start = time.time()
    with SnowflakeClient() as sf:
        write_legends(args.gzip)

        if not args.no_symbols:
            sym_path = OUT / f"symbols{_ext(args.gzip)}"
            if sym_path.exists() and not args.force:
                print(f"\n=== symbols.csv exists, skipping (use --force to refresh) ===")
            else:
                symbols = build_symbols_table(sf)
                _save_csv(symbols, sym_path, args.gzip)

        for g in groups:
            try:
                decode_group(sf, g, since=args.since, gzip=args.gzip, force=args.force)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {g.name} FAILED: {type(e).__name__}: {e}",
                      file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. All CSVs in: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
