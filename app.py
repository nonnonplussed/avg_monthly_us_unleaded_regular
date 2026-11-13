import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

st.set_page_config(page_title="U.S. Regular Gasoline Price Trends", layout="centered")

st.markdown(
    """
    <style>
    /* Mean [slider] Median toggle: show "Mean" in white on the left of the slider, aligned with "Median" */
    .st-key-central_tendency_toggle label {
        display: inline-flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }
    .st-key-central_tendency_toggle label::before {
        content: "Mean";
        color: #FFFFFF !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
    }
    .st-key-central_tendency_toggle label p,
    .st-key-seasonal_events_toggle label p,
    .st-key-annual_events_toggle label p,
    .st-key-ci_toggle label p {
        color: #FFFFFF !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
        margin: 0 !important;
    }

    /* Prefix "Events:" inline before the Seasonal toggle */
    .st-key-seasonal_events_toggle > div {
        display: inline-flex !important;
        flex-direction: row !important;
        align-items: center !important;
    }
    .st-key-seasonal_events_toggle > div::before {
        content: "Events:";
        color: #FFFFFF !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
        margin-right: 0.55rem !important;
        white-space: nowrap !important;
    }

    /* Place "Seasonal" and "Annual" labels before their respective sliders */
    .st-key-seasonal_events_toggle label,
    .st-key-annual_events_toggle label {
        display: inline-flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }
    .st-key-seasonal_events_toggle label > div:last-of-type,
    .st-key-annual_events_toggle label > div:last-of-type {
        order: -1 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* Render view_tab_selector radio as clean underline tabs matching st.tabs */
    .st-key-view_tab_selector [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 1.5rem !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.28) !important;
        padding-bottom: 0 !important;
        margin-bottom: 0.75rem !important;
        width: 100% !important;
    }
    .st-key-view_tab_selector [role="radiogroup"] label {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.45rem 0.1rem 0.55rem 0.1rem !important;
        margin: 0 0 -1px 0 !important;
        cursor: pointer !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
    .st-key-view_tab_selector [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    .st-key-view_tab_selector [role="radiogroup"] label p {
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .st-key-view_tab_selector [role="radiogroup"] label:has(input:checked) {
        border-bottom: 2px solid #ff4b4b !important;
    }
    .st-key-view_tab_selector [role="radiogroup"] label:has(input:checked) p {
        color: #ff4b4b !important;
        font-weight: 600 !important;
    }
    .st-key-view_tab_selector [role="radiogroup"] label:hover p {
        color: #ff4b4b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EIA_HIST_BASE = "https://www.eia.gov/dnav/pet/hist_xls"
EIA_MONTHLY_US_URL = f"{EIA_HIST_BASE}/EMM_EPMR_PTE_NUS_DPGm.xls"
EIA_ANNUAL_US_URL = f"{EIA_HIST_BASE}/EMM_EPMR_PTE_NUS_DPGa.xls"
LOCAL_ANNUAL_CSV = "U.S._Regular_All_Formulations_Retail_Gasoline_Prices.csv"
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"

# EIA State Energy Data System (SEDS) keyless bulk CSV endpoints (all 50 states + D.C.)
SEDS_PRICES_CSV_URL = "https://www.eia.gov/state/seds/sep_prices/total/csv/pr_all.csv"
SEDS_CONVFAC_CSV_URL = "https://www.eia.gov/state/seds/sep_use/total/csv/use_convfac.csv"
SEDS_COMPLETE_CSV_URL = "https://www.eia.gov/state/seds/CDF/Complete_SEDS.csv"

# 9 states with direct monthly & annual EIA Regular-grade retail pump surveys
EIA_DIRECT_STATES = {
    "CA": "SCA",
    "CO": "SCO",
    "FL": "SFL",
    "MA": "SMA",
    "MN": "SMN",
    "NY": "SNY",
    "OH": "SOH",
    "TX": "STX",
    "WA": "SWA",
}

# PADD regions & East Coast sub-districts with direct monthly & annual EIA Regular-grade surveys
PADD_REGIONS = {
    "R10": "PADD 1: East Coast",
    "R1X": "PADD 1A: New England",
    "R1Y": "PADD 1B: Central Atlantic",
    "R1Z": "PADD 1C: Lower Atlantic",
    "R20": "PADD 2: Midwest",
    "R30": "PADD 3: Gulf Coast",
    "R40": "PADD 4: Rocky Mountain",
    "R50": "PADD 5: West Coast",
}

# Every state & D.C. mapped to (Full Name, Parent PADD/Sub-District EIA Code)
STATE_INFO = {
    "AL": ("Alabama", "R30"),
    "AK": ("Alaska", "R50"),
    "AZ": ("Arizona", "R50"),
    "AR": ("Arkansas", "R30"),
    "CA": ("California", "R50"),
    "CO": ("Colorado", "R40"),
    "CT": ("Connecticut", "R1X"),
    "DE": ("Delaware", "R1Y"),
    "DC": ("District of Columbia", "R1Y"),
    "FL": ("Florida", "R1Z"),
    "GA": ("Georgia", "R1Z"),
    "HI": ("Hawaii", "R50"),
    "ID": ("Idaho", "R40"),
    "IL": ("Illinois", "R20"),
    "IN": ("Indiana", "R20"),
    "IA": ("Iowa", "R20"),
    "KS": ("Kansas", "R20"),
    "KY": ("Kentucky", "R20"),
    "LA": ("Louisiana", "R30"),
    "ME": ("Maine", "R1X"),
    "MD": ("Maryland", "R1Y"),
    "MA": ("Massachusetts", "R1X"),
    "MI": ("Michigan", "R20"),
    "MN": ("Minnesota", "R20"),
    "MS": ("Mississippi", "R30"),
    "MO": ("Missouri", "R20"),
    "MT": ("Montana", "R40"),
    "NE": ("Nebraska", "R20"),
    "NV": ("Nevada", "R50"),
    "NH": ("New Hampshire", "R1X"),
    "NJ": ("New Jersey", "R1Y"),
    "NM": ("New Mexico", "R30"),
    "NY": ("New York", "R1Y"),
    "NC": ("North Carolina", "R1Z"),
    "ND": ("North Dakota", "R20"),
    "OH": ("Ohio", "R20"),
    "OK": ("Oklahoma", "R20"),
    "OR": ("Oregon", "R50"),
    "PA": ("Pennsylvania", "R1Y"),
    "RI": ("Rhode Island", "R1X"),
    "SC": ("South Carolina", "R1Z"),
    "SD": ("South Dakota", "R20"),
    "TN": ("Tennessee", "R20"),
    "TX": ("Texas", "R30"),
    "UT": ("Utah", "R40"),
    "VT": ("Vermont", "R1X"),
    "VA": ("Virginia", "R1Z"),
    "WA": ("Washington", "R50"),
    "WV": ("West Virginia", "R1Z"),
    "WI": ("Wisconsin", "R20"),
    "WY": ("Wyoming", "R40"),
}


def build_geography_options() -> tuple[list[str], dict[str, dict]]:
    """Build ordered dropdown labels and metadata lookup for US, PADD regions, and all 50 states + D.C."""
    options = ["United States (National Benchmark)"]
    meta = {
        "United States (National Benchmark)": {
            "kind": "US",
            "code": "NUS",
            "short_name": "U.S.",
            "full_name": "United States",
            "padd": "NUS",
        }
    }

    for r_code, r_label in PADD_REGIONS.items():
        options.append(r_label)
        meta[r_label] = {
            "kind": "PADD",
            "code": r_code,
            "short_name": r_label.split(": ", 1)[1],
            "full_name": r_label,
            "padd": r_code,
        }

    for st_abbr in sorted(STATE_INFO.keys(), key=lambda k: STATE_INFO[k][0]):
        st_name, padd_code = STATE_INFO[st_abbr]
        is_direct = st_abbr in EIA_DIRECT_STATES
        badge = "EIA Direct Survey" if is_direct else f"SEDS + {padd_code}"
        label = f"{st_name} ({st_abbr}) — {badge}"
        options.append(label)
        meta[label] = {
            "kind": "STATE_DIRECT" if is_direct else "STATE_SEDS",
            "code": st_abbr,
            "eia_geo": EIA_DIRECT_STATES.get(st_abbr, padd_code),
            "short_name": st_name,
            "full_name": f"{st_name} ({st_abbr})",
            "padd": padd_code,
        }

    return options, meta


def t_crit_975(df: int) -> float:
    """Two-sided 95% (q=0.975) Student-t critical value with pure-Python fallback."""
    if df <= 0:
        return 1.959963984540054
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df=df))
    except Exception:
        z = 1.959963984540054
        g1 = (z**3 + z) / 4.0
        g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96.0
        g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384.0
        g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160.0
        return z + g1 / df + g2 / (df**2) + g3 / (df**3) + g4 / (df**4)


def mean_ci_95(values: np.ndarray) -> tuple[float, float]:
    """95% Student-t confidence interval for the sample mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    m = float(np.mean(arr))
    if n == 1:
        return (m, m)
    se = float(np.std(arr, ddof=1)) / math.sqrt(n)
    margin = t_crit_975(n - 1) * se
    return (m - margin, m + margin)


def median_ci_95(values: np.ndarray) -> tuple[float, float]:
    """Exact distribution-free 95% confidence interval for the median via Binomial(n, 0.5) order statistics."""
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    half_pow = 0.5 ** n
    cum = 0.0
    lower_idx = 0
    for k in range(n // 2 + 1):
        p_k = math.comb(n, k) * half_pow
        if cum + p_k > 0.025:
            lower_idx = k
            break
        cum += p_k
    upper_idx = max(lower_idx, min(n - 1, n - 1 - lower_idx))
    return (float(arr[lower_idx]), float(arr[upper_idx]))


def _fetch_url_bytes(
    url: str,
    timeout: tuple[float, float] = (3.5, 6.0),
    retries: int = 0,
) -> bytes:
    """Fetch raw bytes from an upstream EIA/FRED endpoint with strict connect/read timeouts."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream,*/*;q=0.8",
    }
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as err:
            last_err = err
    assert last_err is not None
    raise last_err


def _append_ytd_annual_rows(
    annual_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    val_col: str,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Synthesize year-to-date annual rows from monthly data when a new calendar year
    has appeared in monthly surveys before EIA's annual file (DPGa.xls) updates.
    """
    if monthly_df.empty:
        return annual_df, []
    existing_years = set(annual_df["Year"].astype(int)) if not annual_df.empty else set()
    min_ann_year = int(annual_df["Year"].min()) if not annual_df.empty else int(monthly_df["Year"].min())
    ytd_rows: list[dict] = []
    ytd_years: list[int] = []
    for yr, grp in monthly_df.groupby("Year"):
        yr_int = int(yr)
        if yr_int >= min_ann_year and yr_int not in existing_years:
            ytd_rows.append({"Year": yr_int, val_col: float(grp[val_col].mean())})
            ytd_years.append(yr_int)
    if ytd_rows:
        annual_df = (
            pd.concat([annual_df, pd.DataFrame(ytd_rows)], ignore_index=True)
            .sort_values("Year")
            .reset_index(drop=True)
        )
    return annual_df, sorted(ytd_years)


@st.cache_data(ttl=86400, show_spinner="Loading EIA gasoline & BLS CPI data...")
def load_base_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Load CPI-U and U.S. national monthly/annual regular retail gasoline benchmarks concurrently."""
    def _safe_fetch(url: str) -> bytes | None:
        try:
            return _fetch_url_bytes(url, timeout=(3.5, 6.0), retries=0)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_cpi = pool.submit(_safe_fetch, FRED_CPI_URL)
        fut_m = pool.submit(_safe_fetch, EIA_MONTHLY_US_URL)
        fut_a = pool.submit(_safe_fetch, EIA_ANNUAL_US_URL)
        cpi_bytes = fut_cpi.result()
        m_bytes = fut_m.result()
        a_bytes = fut_a.result()

    # 1. U.S. National monthly regular retail gasoline prices (required primary series)
    if m_bytes is None:
        m_bytes = _fetch_url_bytes(EIA_MONTHLY_US_URL, timeout=(4.0, 10.0), retries=1)
    monthly_us = pd.read_excel(io.BytesIO(m_bytes), sheet_name="Data 1", skiprows=2)
    monthly_us.columns = ["Date", "US_Nominal"]
    monthly_us["Date"] = pd.to_datetime(monthly_us["Date"])
    monthly_us["US_Nominal"] = pd.to_numeric(monthly_us["US_Nominal"], errors="coerce")
    monthly_us = (
        monthly_us.dropna(subset=["Date", "US_Nominal"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    monthly_us["YearMonth"] = monthly_us["Date"].dt.to_period("M")
    monthly_us["Year"] = monthly_us["Date"].dt.year

    # 2. Monthly BLS CPI-U from FRED (graceful fallback if FRED stalls/throttles)
    if cpi_bytes is not None:
        cpi = pd.read_csv(io.BytesIO(cpi_bytes), na_values=".")
        cpi.columns = ["Date", "CPI"]
        cpi["Date"] = pd.to_datetime(cpi["Date"])
        cpi = cpi.sort_values("Date").reset_index(drop=True)
        cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="coerce").interpolate(method="linear").ffill().bfill()
        cpi["YearMonth"] = cpi["Date"].dt.to_period("M")
        cpi["Year"] = cpi["Date"].dt.year
    else:
        cpi = monthly_us[["Date", "YearMonth", "Year"]].copy()
        cpi["CPI"] = 100.0

    # Merge CPI after chronological sort so ffill & latest_cpi always reflect the newest month
    monthly_us = pd.merge(monthly_us, cpi[["YearMonth", "CPI"]], on="YearMonth", how="left")
    monthly_us["CPI"] = monthly_us["CPI"].ffill().bfill().fillna(100.0)
    latest_cpi = float(monthly_us["CPI"].iloc[-1])
    monthly_us["US_Real"] = monthly_us["US_Nominal"] * (latest_cpi / monthly_us["CPI"])

    # 3. U.S. National annual regular retail gasoline prices (with local CSV & monthly synthesis fallbacks)
    annual_us = None
    if a_bytes is not None:
        try:
            annual_us = pd.read_excel(io.BytesIO(a_bytes), sheet_name="Data 1", skiprows=2)
            annual_us.columns = ["Date", "US_Nominal"]
            annual_us["Year"] = pd.to_datetime(annual_us["Date"]).dt.year
        except Exception:
            annual_us = None

    if annual_us is None and os.path.exists(LOCAL_ANNUAL_CSV):
        try:
            annual_us = pd.read_csv(LOCAL_ANNUAL_CSV, skiprows=4)
            annual_us.columns = ["Year", "US_Nominal"]
            annual_us["Year"] = pd.to_numeric(annual_us["Year"], errors="coerce").astype(int)
        except Exception:
            annual_us = None

    if annual_us is None:
        annual_us = monthly_us.groupby("Year", as_index=False)["US_Nominal"].mean()

    annual_us["US_Nominal"] = pd.to_numeric(annual_us["US_Nominal"], errors="coerce")
    annual_us = annual_us[["Year", "US_Nominal"]].dropna().sort_values("Year").reset_index(drop=True)
    annual_us, ytd_years = _append_ytd_annual_rows(annual_us, monthly_us, "US_Nominal")

    annual_cpi = cpi.groupby("Year", as_index=False)["CPI"].mean()
    annual_us = pd.merge(annual_us, annual_cpi, on="Year", how="left")
    annual_us["CPI"] = annual_us["CPI"].ffill().bfill().fillna(latest_cpi)
    annual_us["US_Real"] = annual_us["US_Nominal"] * (latest_cpi / annual_us["CPI"])
    annual_us["Is_YTD"] = annual_us["Year"].isin(ytd_years)

    return monthly_us, annual_us, cpi, latest_cpi


@st.cache_data(ttl=86400, show_spinner="Loading regional/state EIA series...")
def load_eia_direct_series(geo_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch monthly and annual Regular-grade XLS series concurrently for a PADD or EIA state."""
    url_m = f"{EIA_HIST_BASE}/EMM_EPMR_PTE_{geo_code}_DPGm.xls"
    url_a = f"{EIA_HIST_BASE}/EMM_EPMR_PTE_{geo_code}_DPGa.xls"

    def _safe_fetch(url: str) -> bytes | None:
        try:
            return _fetch_url_bytes(url, timeout=(3.5, 6.0), retries=0)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_m = pool.submit(_safe_fetch, url_m)
        fut_a = pool.submit(_safe_fetch, url_a)
        m_bytes = fut_m.result()
        a_bytes = fut_a.result()

    if m_bytes is None:
        m_bytes = _fetch_url_bytes(url_m, timeout=(4.0, 10.0), retries=1)

    df_m = pd.read_excel(io.BytesIO(m_bytes), sheet_name="Data 1", skiprows=2)
    df_m.columns = ["Date", "Geo_Nominal"]
    df_m["Date"] = pd.to_datetime(df_m["Date"])
    df_m["Geo_Nominal"] = pd.to_numeric(df_m["Geo_Nominal"], errors="coerce")
    df_m = df_m.dropna(subset=["Date", "Geo_Nominal"]).sort_values("Date").reset_index(drop=True)
    df_m["YearMonth"] = df_m["Date"].dt.to_period("M")
    df_m["Year"] = df_m["Date"].dt.year

    if a_bytes is not None:
        df_a = pd.read_excel(io.BytesIO(a_bytes), sheet_name="Data 1", skiprows=2)
        df_a.columns = ["Date", "Geo_Nominal"]
        df_a["Year"] = pd.to_datetime(df_a["Date"]).dt.year
        df_a["Geo_Nominal"] = pd.to_numeric(df_a["Geo_Nominal"], errors="coerce")
        df_a = df_a[["Year", "Geo_Nominal"]].dropna().sort_values("Year").reset_index(drop=True)
        df_a, _ = _append_ytd_annual_rows(df_a, df_m, "Geo_Nominal")
    else:
        df_a = df_m.groupby("Year", as_index=False)["Geo_Nominal"].mean()

    return df_m[["YearMonth", "Geo_Nominal"]], df_a


@st.cache_data(ttl=86400, show_spinner="Loading EIA SEDS state offsets...")
def load_seds_state_offsets() -> pd.DataFrame:
    """
    Load EIA SEDS annual motor gasoline prices (MGACD, $/MMBtu) and thermal conversion
    factors (MGTCK, MMBtu/gal) for all 50 states + D.C. and US, returning $/gal by StateCode & Year.
    """
    def _read_seds_csv(url: str, msn_filter: set[str]) -> pd.DataFrame:
        content = _fetch_url_bytes(url, timeout=(3.5, 10.0), retries=0)
        raw = pd.read_csv(io.BytesIO(content))
        raw.columns = [c.strip() for c in raw.columns]
        if "MSN" in raw.columns and "StateCode" in raw.columns:
            sub = raw[raw["MSN"].isin(msn_filter)].copy()
            year_cols = [c for c in sub.columns if c.isdigit() and int(c) >= 1990]
            if year_cols:
                melted = sub.melt(
                    id_vars=["MSN", "StateCode"],
                    value_vars=year_cols,
                    var_name="Year",
                    value_name="Val",
                )
                melted["Year"] = melted["Year"].astype(int)
                melted["Val"] = pd.to_numeric(melted["Val"], errors="coerce")
                return melted.dropna(subset=["Val"])
            elif "Year" in sub.columns and "Data" in sub.columns:
                sub = sub[sub["Year"] >= 1990][["MSN", "StateCode", "Year", "Data"]].copy()
                sub.columns = ["MSN", "StateCode", "Year", "Val"]
                sub["Year"] = pd.to_numeric(sub["Year"], errors="coerce").astype(int)
                sub["Val"] = pd.to_numeric(sub["Val"], errors="coerce")
                return sub.dropna(subset=["Val"])
        raise ValueError("Unexpected SEDS CSV column layout")

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_p = pool.submit(_read_seds_csv, SEDS_PRICES_CSV_URL, {"MGACD"})
        fut_c = pool.submit(_read_seds_csv, SEDS_CONVFAC_CSV_URL, {"MGTCK"})
        try:
            prices_long = fut_p.result()
            conv_long = fut_c.result()
            merged = pd.merge(
                prices_long[["StateCode", "Year", "Val"]].rename(columns={"Val": "MGACD"}),
                conv_long[["StateCode", "Year", "Val"]].rename(columns={"Val": "MGTCK"}),
                on=["StateCode", "Year"],
                how="inner",
            )
        except Exception:
            complete = _read_seds_csv(SEDS_COMPLETE_CSV_URL, {"MGACD", "MGTCK"})
            p_df = complete[complete["MSN"] == "MGACD"][["StateCode", "Year", "Val"]].rename(columns={"Val": "MGACD"})
            c_df = complete[complete["MSN"] == "MGTCK"][["StateCode", "Year", "Val"]].rename(columns={"Val": "MGTCK"})
            merged = pd.merge(p_df, c_df, on=["StateCode", "Year"], how="inner")

    merged["SEDS_Price_Gal"] = merged["MGACD"] * merged["MGTCK"]
    return merged[["StateCode", "Year", "SEDS_Price_Gal"]].sort_values(["StateCode", "Year"]).reset_index(drop=True)


def get_geography_series(geo_meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Assemble aligned monthly and annual Nominal & Real price series for the chosen geography:
      - US: Direct EIA national series (NUS)
      - PADD: Direct EIA regional series (R10..R50)
      - STATE_DIRECT (9 states): Direct EIA state series, backfilled pre-start with PADD regional offset
      - STATE_SEDS (other 41 states + D.C.): PADD seasonal profile + time-varying EIA SEDS annual offset
    """
    monthly_us, annual_us, _, latest_cpi = load_base_benchmarks()
    kind = geo_meta["kind"]

    if kind == "US":
        m_out = monthly_us[["Date", "YearMonth", "Year", "CPI", "US_Nominal", "US_Real"]].copy()
        m_out = m_out.rename(columns={"US_Nominal": "Nominal_Price", "US_Real": "Real_Price"})
        m_out["US_Benchmark_Nominal"] = m_out["Nominal_Price"]
        m_out["US_Benchmark_Real"] = m_out["Real_Price"]

        a_out = annual_us[["Year", "CPI", "US_Nominal", "US_Real", "Is_YTD"]].copy()
        a_out = a_out.rename(columns={"US_Nominal": "Nominal_Price", "US_Real": "Real_Price"})
        a_out["US_Benchmark_Nominal"] = a_out["Nominal_Price"]
        a_out["US_Benchmark_Real"] = a_out["Real_Price"]
        return m_out, a_out, "Direct EIA National Regular-Grade Survey (`NUS`)"

    if kind == "PADD":
        padd_code = geo_meta["code"]
        padd_m, padd_a = load_eia_direct_series(padd_code)

        m_out = pd.merge(monthly_us, padd_m, on="YearMonth", how="left")
        m_out["Nominal_Price"] = m_out["Geo_Nominal"].fillna(m_out["US_Nominal"])
        m_out["Real_Price"] = m_out["Nominal_Price"] * (latest_cpi / m_out["CPI"])
        m_out = m_out.rename(columns={"US_Nominal": "US_Benchmark_Nominal", "US_Real": "US_Benchmark_Real"})

        a_out = pd.merge(annual_us, padd_a, on="Year", how="left")
        a_out["Nominal_Price"] = a_out["Geo_Nominal"].fillna(a_out["US_Nominal"])
        a_out["Real_Price"] = a_out["Nominal_Price"] * (latest_cpi / a_out["CPI"])
        a_out = a_out.rename(columns={"US_Nominal": "US_Benchmark_Nominal", "US_Real": "US_Benchmark_Real"})
        return m_out, a_out, f"Direct EIA Regional Regular-Grade Survey (`{padd_code}`)"

    if kind == "STATE_DIRECT":
        st_eia = geo_meta["eia_geo"]
        padd_code = geo_meta["padd"]
        st_m, st_a = load_eia_direct_series(st_eia)
        padd_m, padd_a = load_eia_direct_series(padd_code)

        # Monthly: use exact state survey where available, backfill early-1990s gaps via PADD + earliest spread
        m_out = pd.merge(monthly_us, st_m, on="YearMonth", how="left")
        m_out = pd.merge(m_out, padd_m.rename(columns={"Geo_Nominal": "Padd_Nominal"}), on="YearMonth", how="left")
        overlap_m = m_out.dropna(subset=["Geo_Nominal", "Padd_Nominal"])
        early_spread_m = float((overlap_m["Geo_Nominal"] - overlap_m["Padd_Nominal"]).iloc[:12].mean()) if len(overlap_m) else 0.0
        fallback_m = m_out["Padd_Nominal"].fillna(m_out["US_Nominal"]) + early_spread_m
        m_out["Nominal_Price"] = m_out["Geo_Nominal"].fillna(fallback_m)
        m_out["Real_Price"] = m_out["Nominal_Price"] * (latest_cpi / m_out["CPI"])
        m_out = m_out.rename(columns={"US_Nominal": "US_Benchmark_Nominal", "US_Real": "US_Benchmark_Real"})

        # Annual: exact state survey where available, backfill early-1990s gaps via PADD + earliest annual spread
        a_out = pd.merge(annual_us, st_a, on="Year", how="left")
        a_out = pd.merge(a_out, padd_a.rename(columns={"Geo_Nominal": "Padd_Nominal"}), on="Year", how="left")
        overlap_a = a_out.dropna(subset=["Geo_Nominal", "Padd_Nominal"])
        early_spread_a = float((overlap_a["Geo_Nominal"] - overlap_a["Padd_Nominal"]).iloc[:3].mean()) if len(overlap_a) else 0.0
        fallback_a = a_out["Padd_Nominal"].fillna(a_out["US_Nominal"]) + early_spread_a
        a_out["Nominal_Price"] = a_out["Geo_Nominal"].fillna(fallback_a)
        a_out["Real_Price"] = a_out["Nominal_Price"] * (latest_cpi / a_out["CPI"])
        a_out = a_out.rename(columns={"US_Nominal": "US_Benchmark_Nominal", "US_Real": "US_Benchmark_Real"})

        return m_out, a_out, f"Direct EIA State Regular-Grade Survey (`{st_eia}`, early-1990s backfilled via `{padd_code}`)"

    # STATE_SEDS: Combine PADD regional Regular-grade seasonality with EIA SEDS state annual offsets
    st_abbr = geo_meta["code"]
    padd_code = geo_meta["padd"]
    padd_m, padd_a = load_eia_direct_series(padd_code)

    try:
        seds = load_seds_state_offsets()
        st_seds = seds[seds["StateCode"] == st_abbr][["Year", "SEDS_Price_Gal"]].rename(columns={"SEDS_Price_Gal": "St_SEDS"})
        us_seds = seds[seds["StateCode"] == "US"][["Year", "SEDS_Price_Gal"]].rename(columns={"SEDS_Price_Gal": "US_SEDS"})
        seds_offsets = pd.merge(st_seds, us_seds, on="Year", how="inner")
        seds_offsets["State_vs_US_Offset"] = seds_offsets["St_SEDS"] - seds_offsets["US_SEDS"]
        has_seds = len(seds_offsets) > 0
    except Exception:
        seds_offsets = pd.DataFrame(columns=["Year", "State_vs_US_Offset"])
        has_seds = False

    # Annual series: US Regular + time-varying SEDS State-vs-US offset (carry forward latest SEDS year for 2025–2026)
    a_out = pd.merge(annual_us, padd_a.rename(columns={"Geo_Nominal": "Padd_Nominal"}), on="Year", how="left")
    if has_seds:
        a_out = pd.merge(a_out, seds_offsets[["Year", "State_vs_US_Offset"]], on="Year", how="left")
        a_out["State_vs_US_Offset"] = a_out["State_vs_US_Offset"].ffill().bfill().fillna(0.0)
        a_out["Nominal_Price"] = a_out["US_Nominal"] + a_out["State_vs_US_Offset"]
    else:
        a_out["Nominal_Price"] = a_out["Padd_Nominal"].fillna(a_out["US_Nominal"])
    a_out["Real_Price"] = a_out["Nominal_Price"] * (latest_cpi / a_out["CPI"])
    a_out = a_out.rename(columns={"US_Nominal": "US_Benchmark_Nominal", "US_Real": "US_Benchmark_Real"})

    # Monthly series: PADD regional Regular-grade monthly series + annual state-vs-PADD residual offset
    padd_ann_avg = padd_a.rename(columns={"Geo_Nominal": "Padd_Ann_Nominal"})
    state_vs_padd_yr = pd.merge(a_out[["Year", "Nominal_Price"]], padd_ann_avg, on="Year", how="left")
    state_vs_padd_yr["State_vs_Padd_Offset"] = (
        state_vs_padd_yr["Nominal_Price"] - state_vs_padd_yr["Padd_Ann_Nominal"]
    ).fillna(0.0)

    m_out = pd.merge(monthly_us, padd_m.rename(columns={"Geo_Nominal": "Padd_Nominal"}), on="YearMonth", how="left")
    m_out = pd.merge(m_out, state_vs_padd_yr[["Year", "State_vs_Padd_Offset"]], on="Year", how="left")
    m_out["State_vs_Padd_Offset"] = m_out["State_vs_Padd_Offset"].ffill().bfill().fillna(0.0)
    m_out["Nominal_Price"] = m_out["Padd_Nominal"].fillna(m_out["US_Nominal"]) + m_out["State_vs_Padd_Offset"]
    m_out["Real_Price"] = m_out["Nominal_Price"] * (latest_cpi / m_out["CPI"])
    m_out = m_out.rename(columns={"US_Nominal": "US_Benchmark_Nominal", "US_Real": "US_Benchmark_Real"})

    source_note = (
        f"EIA SEDS Annual State Offset (`{st_abbr}` vs. `US`) + `{padd_code}` ({PADD_REGIONS[padd_code].split(': ')[1]}) Seasonal Profile"
        if has_seds
        else f"EIA Regional Proxy (`{padd_code}` — {PADD_REGIONS[padd_code].split(': ')[1]})"
    )
    return m_out, a_out, source_note


def _fmt_dollar(v: float) -> str:
    return "—" if pd.isna(v) else f"${v:.2f}"


def _fmt_signed_dollar(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:+.2f}"


def _fmt_signed_pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:+.1f}%"


# ==============================================================================
# TOP CONTROL BAR: GEOGRAPHY DROPDOWN, PRICE BASIS, CENTRAL TENDENCY & OVERLAYS
# ==============================================================================
geo_options, geo_meta_map = build_geography_options()
base_monthly_us, base_annual_us, _, _ = load_base_benchmarks()
latest_label = f"{base_monthly_us['Date'].max():%B %Y}"

m_min_yr_global = int(base_monthly_us["Year"].min())
m_max_yr_global = int(base_monthly_us["Year"].max())
all_seasonal_label_global = f"{m_min_yr_global}–{m_max_yr_global} (All)"

seasonal_tab_label = "Seasonal Trends (By Month)"
annual_tab_label = f"Annual Trends ({int(base_annual_us['Year'].min())}–{int(base_annual_us['Year'].max())})"

# Evaluate active view and seasonal year prior to rendering 95% CI toggle
active_tab_pre = st.session_state.get("view_tab_selector", seasonal_tab_label)
selected_seasonal_yr_pre = st.session_state.get("seasonal_year_select", all_seasonal_label_global)
is_single_seasonal_yr_pre = selected_seasonal_yr_pre != all_seasonal_label_global
ci_locked_off = (active_tab_pre == seasonal_tab_label) and is_single_seasonal_yr_pre

if "user_ci_pref" not in st.session_state:
    st.session_state["user_ci_pref"] = True

if ci_locked_off:
    st.session_state["ci_toggle"] = False
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(.st-key-ci_toggle),
        div[data-testid="stColumn"]:has(.st-key-ci_toggle),
        div[data-testid="stVerticalBlock"]:has(.st-key-ci_toggle),
        .st-key-ci_toggle {
            overflow: visible !important;
            position: relative !important;
            z-index: 9999 !important;
        }
        .st-key-ci_toggle {
            pointer-events: auto !important;
            cursor: not-allowed !important;
        }
        .st-key-ci_toggle * {
            cursor: not-allowed !important;
        }
        .st-key-ci_toggle:hover::after {
            content: "95% CI requires All years in Seasonal view";
            position: absolute;
            top: calc(100% + 8px);
            right: 0;
            background-color: #262730;
            color: #FFFFFF;
            border: 1px solid rgba(250, 250, 250, 0.2);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.78rem;
            line-height: 1.35;
            white-space: nowrap;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    if st.session_state.get("_was_ci_locked_off", False):
        st.session_state["ci_toggle"] = st.session_state["user_ci_pref"]
    elif "ci_toggle" not in st.session_state:
        st.session_state["ci_toggle"] = st.session_state["user_ci_pref"]

st.session_state["_was_ci_locked_off"] = ci_locked_off

top_col1, top_col2 = st.columns([1.35, 1.65])
with top_col1:
    selected_geo_label = st.selectbox(
        "Geography (U.S., PADD Region, or State):",
        options=geo_options,
        index=0,
        key="geo_select",
        help="Type to search any of the 50 states + D.C., 8 PADD regions, or the U.S. national benchmark",
    )
with top_col2:
    mode = st.radio(
        "Price Basis:",
        options=[f"Inflation-Adjusted ({latest_label} Dollars)", "Nominal Dollars (Unadjusted)"],
        index=0,
        horizontal=False,
    )

sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(
    [1.26, 0.86, 1.14, 0.94],
    gap="small",
    vertical_alignment="center",
)
with sub_col1:
    show_seasonal_events = st.toggle(
        "Seasonal",
        value=True,
        key="seasonal_events_toggle",
        help="Blend transitions (Mar, Nov) & summer driving season (May–Aug)",
    )
with sub_col2:
    show_annual_events = st.toggle(
        "Annual",
        value=True,
        key="annual_events_toggle",
        help="Hurricanes, macro shocks, OPEC/shale, geopolitics & SPR releases",
    )
with sub_col3:
    is_median = st.toggle(
        "Median",
        value=True,
        key="central_tendency_toggle",
        help="Left: Mean · Right: Median",
    )
with sub_col4:
    show_ci = st.toggle(
        "95% CI",
        key="ci_toggle",
        disabled=ci_locked_off,
        help=(
            "95% CI requires All years in Seasonal view"
            if ci_locked_off
            else "Adds the 95% confidence interval for the selected central tendency (Mean or Median)"
        ),
    )
    if not ci_locked_off:
        st.session_state["user_ci_pref"] = show_ci

selected_meta = geo_meta_map[selected_geo_label]
is_us_benchmark = selected_meta["kind"] == "US"
geo_short = selected_meta["short_name"]
monthly_df, annual_df, geo_method_note = get_geography_series(selected_meta)
for _df in (monthly_df, annual_df):
    for _col in ("Nominal_Price", "Real_Price", "US_Benchmark_Nominal", "US_Benchmark_Real"):
        _df[_col] = _df[_col].round(2)

is_real = mode.startswith("Inflation-Adjusted")
price_col = "Real_Price" if is_real else "Nominal_Price"
us_bench_col = "US_Benchmark_Real" if is_real else "US_Benchmark_Nominal"
agg_func = "median" if is_median else "mean"
ci_func = median_ci_95 if is_median else mean_ci_95
stat_noun = "Median" if is_median else "Mean"
y_label_seasonal = f"{stat_noun} Price ({latest_label} $)" if is_real else f"{stat_noun} Price ($)"
y_label_annual = f"{stat_noun} Price ({latest_label} $)" if is_real else f"{stat_noun} Price ($)"

# Compute summary offset vs. U.S. national benchmark for non-US geographies
if not is_us_benchmark:
    avg_offset_dollar = float((annual_df[price_col] - annual_df[us_bench_col]).mean())
    avg_offset_dollar = round(avg_offset_dollar, 2)
    avg_offset_pct = float(((annual_df[price_col].mean() / annual_df[us_bench_col].mean()) - 1.0) * 100)
    st.caption(
        f"**Active Series:** {selected_meta['full_name']} · **Source:** {geo_method_note}  \n"
        f"**Avg Offset vs. U.S.:** `{avg_offset_dollar:+.2f}/gal` (`{avg_offset_pct:+.1f}%`)"
    )

active_view_tab = st.radio(
    "View:",
    options=[seasonal_tab_label, annual_tab_label],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="view_tab_selector",
)

# ==============================================================================
# TAB 1: SEASONAL TRENDS BY CALENDAR MONTH
# ==============================================================================
if active_view_tab == seasonal_tab_label:
    month_names = ["01 Jan", "02 Feb", "03 Mar", "04 Apr", "05 May", "06 Jun", "07 Jul", "08 Aug", "09 Sep", "10 Oct", "11 Nov", "12 Dec"]
    month_full_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    viridis_hex = [
        "#440154", "#482173", "#433e85", "#38588c", "#2d708e", "#25858e",
        "#1e9b8a", "#2ab07f", "#52c569", "#86d549", "#c2df23", "#fde725",
    ]

    m_min_yr = int(monthly_df["Year"].min())
    m_max_yr = int(monthly_df["Year"].max())
    all_seasonal_label = f"{m_min_yr}–{m_max_yr} (All)"
    seasonal_year_options = [all_seasonal_label] + [str(y) for y in range(m_max_yr, m_min_yr - 1, -1)]

    s_yr_col, _ = st.columns([1.25, 2.75])
    with s_yr_col:
        selected_seasonal_year = st.selectbox(
            "Year to Display:",
            options=seasonal_year_options,
            index=0,
            key="seasonal_year_select",
            help=f"Select '{all_seasonal_label}' for the multi-year seasonal profile or an individual year ({m_min_yr}–{m_max_yr})",
        )

    is_single_seasonal_year = selected_seasonal_year != all_seasonal_label
    if is_single_seasonal_year:
        sel_yr_m = int(selected_seasonal_year)
        seasonal_src = monthly_df[monthly_df["Year"] == sel_yr_m].copy()
    else:
        sel_yr_m = None
        seasonal_src = monthly_df.copy()

    seasonal_src["Month_Num"] = seasonal_src["Date"].dt.month
    summary = seasonal_src.groupby("Month_Num")[price_col].agg(Count="count", Value=agg_func).reindex(range(1, 13))
    summary["Count"] = summary["Count"].fillna(0).astype(int)
    us_month_summary = seasonal_src.groupby("Month_Num")[us_bench_col].agg(US_Val=agg_func).reindex(range(1, 13))
    summary["US_Value"] = us_month_summary["US_Val"]
    summary["Value"] = summary["Value"].round(2)
    summary["US_Value"] = summary["US_Value"].round(2)
    summary["Offset_vs_US"] = (summary["Value"] - summary["US_Value"]).round(2)
    summary["Month"] = month_names

    ci_lows, ci_highs, ci_fmts = [], [], []
    max_top_both = 0.0
    for m_idx in range(1, 13):
        m_vals = seasonal_src.loc[seasonal_src["Month_Num"] == m_idx, price_col].dropna().to_numpy()
        n_obs = len(m_vals)
        if n_obs == 0:
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)
            ci_fmts.append("—")
        else:
            lo_mean, hi_mean = mean_ci_95(m_vals)
            lo_med, hi_med = median_ci_95(m_vals)
            max_top_both = max(max_top_both, hi_mean, hi_med)
            lo_sel, hi_sel = (lo_med, hi_med) if is_median else (lo_mean, hi_mean)
            lo_r, hi_r = round(lo_sel, 2), round(hi_sel, 2)
            ci_lows.append(lo_r)
            ci_highs.append(hi_r)
            ci_fmts.append("N/A (n=1)" if n_obs == 1 else f"${lo_r:.2f} – ${hi_r:.2f}")

    summary["CI_Low"] = ci_lows
    summary["CI_High"] = ci_highs
    summary["CI_Fmt"] = ci_fmts

    first_valid_m = summary["Value"].first_valid_index()
    if first_valid_m is not None:
        base_month_full = month_full_names[first_valid_m - 1]
        base_m_val = summary.loc[first_valid_m, "Value"]
        summary["Change_vs_Jan"] = (summary["Value"] - base_m_val).round(2)
        summary["Pct_Change_vs_Jan"] = (summary["Change_vs_Jan"] / base_m_val) * 100
    else:
        base_month_full = "January"
        summary["Change_vs_Jan"] = np.nan
        summary["Pct_Change_vs_Jan"] = np.nan

    if is_single_seasonal_year:
        prev_val = summary["Value"].shift(1)
        summary["MoM_Pct"] = ((summary["Value"] - prev_val) / prev_val) * 100
    else:
        prev_val = summary["Value"].shift(1)
        prev_val.iloc[0] = summary["Value"].iloc[-1]
        summary["MoM_Pct"] = ((summary["Value"] - prev_val) / prev_val) * 100

    summary["Val_Price_Fmt"] = summary["Value"].map(_fmt_dollar)
    summary["Offset_US_Fmt"] = summary["Offset_vs_US"].map(_fmt_signed_dollar)
    summary["Change_Jan_Fmt"] = summary["Change_vs_Jan"].map(_fmt_signed_dollar)
    summary["Pct_Jan_Fmt"] = summary["Pct_Change_vs_Jan"].map(_fmt_signed_pct)
    summary["MoM_Pct_Fmt"] = summary["MoM_Pct"].map(_fmt_signed_pct)

    min_y_floor = 2.0 if is_single_seasonal_year else 3.25
    y_max_m = float(max(min_y_floor, np.ceil(max_top_both * 2) / 2 + 0.55))
    y_grid_vals = [round(float(v), 1) for v in np.arange(0, y_max_m + 0.01, 0.5)]
    locked_y_scale_m = alt.Scale(domain=[0, y_max_m], nice=False, clamp=True)

    date_span = f"{seasonal_src['Date'].min():%B %Y} - {seasonal_src['Date'].max():%B %Y}"
    subtitle_parts_m = [
        f"{date_span} (Inflation-Adjusted to {latest_label} Dollars)" if is_real else date_span
    ]
    if show_ci and not is_single_seasonal_year:
        subtitle_parts_m.append(f"Error bars = 95% CI ({stat_noun.lower()})")
    subtitle_m = " · ".join(subtitle_parts_m)

    seasonal_records = summary[
        [
            "Month",
            "Count",
            "Value",
            "CI_Low",
            "CI_High",
            "CI_Fmt",
            "Val_Price_Fmt",
            "Offset_US_Fmt",
            "Change_Jan_Fmt",
            "Pct_Jan_Fmt",
            "MoM_Pct_Fmt",
        ]
    ].to_dict(orient="records")
    valid_bar_records = [r for r in seasonal_records if r["Value"] is not None and not pd.isna(r["Value"])]

    seasonal_tooltips = [
        alt.Tooltip("Month:N", title="Month"),
        alt.Tooltip("Val_Price_Fmt:N", title=f"{geo_short} {stat_noun} Price"),
    ]
    if not is_us_benchmark:
        seasonal_tooltips.append(alt.Tooltip("Offset_US_Fmt:N", title="Offset vs. U.S. ($)"))
    if show_ci:
        seasonal_tooltips.append(alt.Tooltip("CI_Fmt:N", title=f"95% CI ({stat_noun})"))
    seasonal_tooltips.extend([
        alt.Tooltip("Change_Jan_Fmt:N", title=f"Change vs. {base_month_full}"),
        alt.Tooltip("Pct_Jan_Fmt:N", title=f"% Change vs. {base_month_full}"),
        alt.Tooltip("MoM_Pct_Fmt:N", title="% Change vs. Prev. Month"),
        alt.Tooltip("Count:Q", title="Observations"),
    ])

    seasonal_base = (
        alt.Chart(alt.Data(values=valid_bar_records))
        .mark_bar()
        .encode(
            x=alt.X(
                "Month:N",
                sort=month_names,
                title="Month",
                scale=alt.Scale(domain=month_names, paddingInner=0.16, paddingOuter=0.12),
                axis=alt.Axis(
                    labelAngle=0,
                    labelFontSize=11,
                    titleFontSize=13,
                    titlePadding=8,
                    labelColor="#333333",
                    titleColor="#111111",
                    grid=True,
                    gridColor="#EAEAEA",
                ),
            ),
            y=alt.Y(
                "Value:Q",
                title=y_label_seasonal,
                scale=locked_y_scale_m,
                axis=alt.Axis(
                    values=y_grid_vals,
                    labelExpr="datum.value % 1 === 0 ? datum.value : ''",
                    grid=True,
                    gridColor="#EAEAEA",
                    labelFontSize=12,
                    titleFontSize=13,
                    titlePadding=8,
                    minExtent=36,
                    labelColor="#333333",
                    titleColor="#111111",
                ),
            ),
            color=alt.Color(
                "Month:N",
                sort=month_names,
                scale=alt.Scale(domain=month_names, range=viridis_hex),
                legend=alt.Legend(
                    title="month",
                    orient="right",
                    titleFontSize=13,
                    labelFontSize=11.5,
                    symbolType="square",
                    symbolSize=220,
                    labelColor="#333333",
                    titleColor="#111111",
                ),
            ),
            tooltip=seasonal_tooltips,
        )
    )

    seasonal_layers = [seasonal_base]

    if show_ci and not is_single_seasonal_year:
        ci_records = [r for r in seasonal_records if r["Count"] > 1 and not pd.isna(r["CI_Low"])]
        ci_data = alt.Data(values=ci_records)
        ci_stem_halo = (
            alt.Chart(ci_data)
            .mark_rule(color="white", strokeWidth=3.6, opacity=0.85)
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("CI_Low:Q", scale=locked_y_scale_m),
                y2=alt.Y2("CI_High:Q"),
            )
        )
        ci_stem = (
            alt.Chart(ci_data)
            .mark_rule(color="#111111", strokeWidth=1.8)
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("CI_Low:Q", scale=locked_y_scale_m),
                y2=alt.Y2("CI_High:Q"),
                tooltip=[
                    alt.Tooltip("Month:N", title="Month"),
                    alt.Tooltip("Val_Price_Fmt:N", title=f"{geo_short} {stat_noun} Price"),
                    alt.Tooltip("CI_Fmt:N", title=f"95% CI ({stat_noun})"),
                ],
            )
        )
        ci_cap_top_halo = (
            alt.Chart(ci_data)
            .mark_tick(color="white", thickness=3.6, size=14, opacity=0.85)
            .encode(x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)), y=alt.Y("CI_High:Q", scale=locked_y_scale_m))
        )
        ci_cap_top = (
            alt.Chart(ci_data)
            .mark_tick(color="#111111", thickness=1.8, size=12)
            .encode(x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)), y=alt.Y("CI_High:Q", scale=locked_y_scale_m))
        )
        ci_cap_bot_halo = (
            alt.Chart(ci_data)
            .mark_tick(color="white", thickness=3.6, size=14, opacity=0.85)
            .encode(x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)), y=alt.Y("CI_Low:Q", scale=locked_y_scale_m))
        )
        ci_cap_bot = (
            alt.Chart(ci_data)
            .mark_tick(color="#111111", thickness=1.8, size=12)
            .encode(x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)), y=alt.Y("CI_Low:Q", scale=locked_y_scale_m))
        )
        seasonal_layers.extend([
            ci_stem_halo,
            ci_stem,
            ci_cap_top_halo,
            ci_cap_top,
            ci_cap_bot_halo,
            ci_cap_bot,
        ])

    if show_seasonal_events:
        top_ref_col = "CI_High" if (show_ci and not is_single_seasonal_year) else "Value"
        top_lookup = dict(zip(summary["Month"], summary[top_ref_col]))
        max_top_current = float(summary[top_ref_col].dropna().max()) if summary[top_ref_col].notna().any() else 0.0
        headroom = max(0.35, y_max_m - max_top_current)

        y_summer_band = max_top_current + headroom * 0.64
        y_blend_marker = max_top_current + headroom * 0.30

        def _anchor(m_str: str) -> float:
            val = top_lookup.get(m_str, np.nan)
            return float(val) if pd.notna(val) else 0.0

        seasonal_event_points = [
            {
                "Month": month_names[2],
                "anchor_top": _anchor(month_names[2]),
                "y_pos": y_blend_marker,
                "Event_Type": "Monthly Event 1: Fuel Blend Transitions & Maintenance",
                "Tag": "Summer-Blend Switch",
                "Detail": "EPA summer-blend transition & spring refinery maintenance (sharpest monthly rise)",
            },
            {
                "Month": month_names[10],
                "anchor_top": _anchor(month_names[10]),
                "y_pos": y_blend_marker,
                "Event_Type": "Monthly Event 1: Fuel Blend Transitions & Maintenance",
                "Tag": "Winter-Blend Switch",
                "Detail": "Autumn switch back to lower-cost winter-blend gasoline (sharpest monthly drop)",
            },
            {
                "Month": month_names[4],
                "anchor_top": _anchor(month_names[4]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Summer driving season begins around Memorial Day",
            },
            {
                "Month": month_names[5],
                "anchor_top": _anchor(month_names[5]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Peak summer highway travel demand & summer-grade fuel plateau",
            },
            {
                "Month": month_names[6],
                "anchor_top": _anchor(month_names[6]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Mid-summer travel demand keeps refinery utilization elevated",
            },
            {
                "Month": month_names[7],
                "anchor_top": _anchor(month_names[7]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Late-summer travel demand through Labor Day",
            },
        ]

        leader_lines = (
            alt.Chart(alt.Data(values=seasonal_event_points))
            .mark_rule(color="#888888", strokeDash=[2, 2], strokeWidth=1.1)
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("anchor_top:Q", scale=locked_y_scale_m),
                y2=alt.Y2("y_pos:Q"),
            )
        )

        span_records = [
            {"Month": m, "span_y": y_summer_band, "Detail": "Monthly Event 2: Summer Driving Season Demand (May–Aug)"}
            for m in month_names[4:8]
        ]

        span_line = (
            alt.Chart(alt.Data(values=span_records))
            .mark_line(color="#d95f02", strokeWidth=2.3, strokeDash=[5, 3])
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("span_y:Q", scale=locked_y_scale_m),
                tooltip=[alt.Tooltip("Detail:N", title="Seasonal Driver")],
            )
        )

        span_label = (
            alt.Chart(alt.Data(values=[{"Month": month_names[5], "span_y": y_summer_band, "Text": "Summer Driving Season (May–Aug)"}]))
            .mark_text(dy=-11, dx=22, fontSize=10.5, fontWeight="bold", color="#111111")
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("span_y:Q", scale=locked_y_scale_m),
                text="Text:N",
            )
        )

        event_markers = (
            alt.Chart(alt.Data(values=seasonal_event_points))
            .mark_point(shape="diamond", size=135, filled=True, stroke="white", strokeWidth=1.2)
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("y_pos:Q", scale=locked_y_scale_m),
                fill=alt.Fill(
                    "Event_Type:N",
                    scale=alt.Scale(
                        domain=[
                            "Monthly Event 1: Fuel Blend Transitions & Maintenance",
                            "Monthly Event 2: Summer Driving Season Demand",
                        ],
                        range=["#e7298a", "#d95f02"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Month:N", title="Month"),
                    alt.Tooltip("Event_Type:N", title="Category"),
                    alt.Tooltip("Detail:N", title="Mechanism"),
                ],
            )
        )

        event_labels = (
            alt.Chart(alt.Data(values=[r for r in seasonal_event_points if r["Tag"]]))
            .mark_text(dy=-11, fontSize=10.5, fontWeight="bold", color="#111111")
            .encode(
                x=alt.X("Month:N", sort=month_names, scale=alt.Scale(domain=month_names)),
                y=alt.Y("y_pos:Q", scale=locked_y_scale_m),
                text="Tag:N",
            )
        )

        seasonal_layers.extend([leader_lines, span_line, span_label, event_markers, event_labels])

    seasonal_title_prefix = f"{geo_short} {stat_noun}" if not is_us_benchmark else stat_noun
    seasonal_chart_title = (
        f"{seasonal_title_prefix} Price of Unleaded Regular Gasoline by Month ({sel_yr_m})"
        if is_single_seasonal_year
        else f"{seasonal_title_prefix} Price of Unleaded Regular Gasoline by Month of the Year"
    )
    seasonal_chart = (
        alt.layer(*seasonal_layers)
        .resolve_scale(color="independent", y="shared")
        .properties(
            height=460,
            title=alt.TitleParams(
                text=seasonal_chart_title,
                subtitle=subtitle_m,
                anchor="start",
                fontSize=17,
                subtitleFontSize=13,
                color="#111111",
                subtitleColor="#333333",
                offset=14,
            ),
        )
        .configure_view(stroke="#333333", strokeWidth=0.8, fill="white")
        .configure(background="white", padding={"top": 20, "bottom": 18, "left": 14, "right": 14})
    )

    st.altair_chart(seasonal_chart, width="stretch")

    seasonal_table_dict = {
        "Month": summary["Month"],
        "Observations": summary["Count"],
        f"{geo_short} {stat_noun} Price": summary["Val_Price_Fmt"],
    }
    if not is_us_benchmark:
        seasonal_table_dict["Offset vs. U.S. ($)"] = summary["Offset_US_Fmt"]
    if show_ci:
        seasonal_table_dict[f"95% CI ({stat_noun})"] = summary["CI_Fmt"]
    seasonal_table_dict.update({
        f"Change vs. {base_month_full}": summary["Change_Jan_Fmt"],
        f"% Change vs. {base_month_full}": summary["Pct_Jan_Fmt"],
        "% Change vs. Prev. Month": summary["MoM_Pct_Fmt"],
    })
    st.dataframe(pd.DataFrame(seasonal_table_dict), height="content", width="stretch", hide_index=True)

    _kind = selected_meta["kind"]
    _padd_code = selected_meta["padd"]
    _us_m_link = "[U.S. EIA Retail Gasoline Prices (`NUS`)](https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_m.htm)"
    _cpi_link = f"[BLS CPI-U via FRED (`CPIAUCSL`)]({FRED_CPI_URL})"

    if _kind == "US":
        _source_links = f"{_us_m_link} & {_cpi_link}"
    elif _kind == "PADD":
        _padd_url = f"https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_{_padd_code.lower()}_m.htm"
        _source_links = (
            f"[EIA {geo_short} Retail Gasoline Prices (`{_padd_code}`)]({_padd_url}), "
            f"{_us_m_link}, & {_cpi_link}"
        )
    elif _kind == "STATE_DIRECT":
        _st_eia = selected_meta["eia_geo"]
        _st_url = f"https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_{_st_eia.lower()}_m.htm"
        _padd_url = f"https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_{_padd_code.lower()}_m.htm"
        _source_links = (
            f"[EIA {geo_short} Retail Gasoline Prices (`{_st_eia}`)]({_st_url}), "
            f"[EIA `{_padd_code}` Backfill Series]({_padd_url}), "
            f"{_us_m_link}, & {_cpi_link}"
        )
    else:
        _padd_name = PADD_REGIONS[_padd_code].split(": ", 1)[1]
        _padd_url = f"https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_{_padd_code.lower()}_m.htm"
        _source_links = (
            f"[EIA SEDS State Prices (`pr_all.csv`)]({SEDS_PRICES_CSV_URL}), "
            f"[EIA SEDS Conversion Factors (`use_convfac.csv`)]({SEDS_CONVFAC_CSV_URL}), "
            f"[EIA {_padd_name} (`{_padd_code}`) Seasonal Series]({_padd_url}), "
            f"{_us_m_link}, & {_cpi_link}"
        )

    st.caption(f"Sources: {_source_links}, {date_span}.")

# ==============================================================================
# TAB 2: ANNUAL TRENDS (1992–PRESENT)
# ==============================================================================
else:
    ann = annual_df.copy()
    ann["Year_Str"] = ann["Year"].astype(str)

    if is_median:
        yr_medians = (
            monthly_df.groupby("Year", as_index=False)[[price_col, us_bench_col]]
            .median()
            .rename(columns={price_col: "Med_Price", us_bench_col: "Med_US_Price"})
        )
        ann = pd.merge(ann, yr_medians, on="Year", how="left")
        ann["Price"] = ann["Med_Price"].fillna(ann[price_col]).round(2)
        ann["US_Price"] = ann["Med_US_Price"].fillna(ann[us_bench_col]).round(2)
    else:
        ann["Price"] = ann[price_col].round(2)
        ann["US_Price"] = ann[us_bench_col].round(2)

    a_min_yr = int(ann["Year"].min())
    a_max_yr = int(ann["Year"].max())
    all_annual_label = f"{a_min_yr}–{a_max_yr} (All)"
    annual_year_options = [all_annual_label] + [str(y) for y in range(a_max_yr, a_min_yr - 1, -1)]

    a_yr_col, _ = st.columns([1.25, 2.75])
    with a_yr_col:
        selected_annual_year = st.selectbox(
            "Year to Display:",
            options=annual_year_options,
            index=0,
            key="annual_year_select",
            help=f"Select '{all_annual_label}' for the full historical series or spotlight an individual year ({a_min_yr}–{a_max_yr})",
        )

    is_single_annual_year = selected_annual_year != all_annual_label
    sel_yr_a = int(selected_annual_year) if is_single_annual_year else None

    ann["Offset_vs_US"] = (ann["Price"] - ann["US_Price"]).round(2)
    ann["YoY_Diff"] = ann["Price"].diff().round(2)
    ann["YoY_Pct"] = ann["Price"].pct_change() * 100

    start_year = int(ann["Year"].iloc[0])
    end_year = int(ann["Year"].iloc[-1])
    base_price = float(ann["Price"].iloc[0])
    end_price = float(ann["Price"].iloc[-1])
    ann["Pct_vs_Start"] = ((ann["Price"] - base_price) / base_price) * 100

    ann["Price_Fmt"] = ann["Price"].map("${:.2f}".format)
    ann["US_Price_Fmt"] = ann["US_Price"].map("${:.2f}".format)
    ann["Offset_US_Fmt"] = ann["Offset_vs_US"].map("{:+.2f}".format)
    ann["YoY_Diff_Fmt"] = ann["YoY_Diff"].map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}")
    ann["YoY_Pct_Fmt"] = ann["YoY_Pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%")
    ann["Pct_Start_Fmt"] = ann["Pct_vs_Start"].map("{:+.1f}%".format)
    ann["Direction"] = np.where(ann["YoY_Pct"].fillna(0) >= 0, "Increase", "Decrease")
    ann["Is_Selected"] = (ann["Year"] == sel_yr_a) if is_single_annual_year else True

    ann_prices_arr = ann["Price"].to_numpy()
    central_ann = float(np.median(ann_prices_arr)) if is_median else float(np.mean(ann_prices_arr))
    ci_low_ann, ci_high_ann = ci_func(ann_prices_arr)
    central_ann = round(central_ann, 2)
    ci_low_ann, ci_high_ann = round(ci_low_ann, 2), round(ci_high_ann, 2)
    ci_ann_fmt = f"${ci_low_ann:.2f}–${ci_high_ann:.2f}"

    central_word_title = "Median" if is_median else "Mean"
    central_word_lower = "median" if is_median else "mean"

    peak_row = ann.loc[ann["Price"].idxmax()]
    low_row = ann.loc[ann["Price"].idxmin()]
    span_years = max(1, end_year - start_year)
    cagr = ((end_price / base_price) ** (1.0 / span_years) - 1.0) * 100

    if not is_single_annual_year:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"{len(ann)}-Year {central_word_title}",
            f"${central_ann:.2f}",
            f"95% CI: {ci_ann_fmt}" if show_ci else f"{start_year}–{end_year} {central_word_lower}",
        )
        col2.metric(
            f"Peak Year ({int(peak_row['Year'])})",
            f"${peak_row['Price']:.2f}",
            f"{peak_row['Pct_vs_Start']:+.1f}% vs. {start_year}",
        )
        col3.metric(
            f"Lowest Year ({int(low_row['Year'])})",
            f"${low_row['Price']:.2f}",
            f"{low_row['Pct_vs_Start']:+.1f}% vs. {start_year}",
        )
        col4.metric(
            "Annualized Growth",
            f"{cagr:+.1f}%/yr",
            f"${base_price:.2f} → ${end_price:.2f}",
        )
    else:
        sel_row = ann.loc[ann["Year"] == sel_yr_a].iloc[0]
        sel_price = float(sel_row["Price"])
        diff_vs_central = round(sel_price - central_ann, 2)
        pct_vs_central = ((sel_price - central_ann) / central_ann) * 100
        rank_pos = int((ann["Price"] > sel_price).sum()) + 1

        yr_m_sub = monthly_df[monthly_df["Year"] == sel_yr_a].copy()
        if len(yr_m_sub) > 0:
            yr_m_sub["Month_Name"] = yr_m_sub["Date"].dt.strftime("%b")
            m_peak = yr_m_sub.loc[yr_m_sub[price_col].idxmax()]
            m_low = yr_m_sub.loc[yr_m_sub[price_col].idxmin()]
            yr_ci_lo, yr_ci_hi = ci_func(yr_m_sub[price_col].to_numpy())
            yr_ci_lo, yr_ci_hi = round(yr_ci_lo, 2), round(yr_ci_hi, 2)
            c1_sub = (
                f"95% CI ({len(yr_m_sub)} mos): ${yr_ci_lo:.2f}–${yr_ci_hi:.2f}"
                if show_ci
                else f"YoY: {sel_row['YoY_Diff_Fmt']} ({sel_row['YoY_Pct_Fmt']})"
            )
            c3_label = f"{sel_yr_a} Peak ({m_peak['Month_Name']})"
            c3_val = f"${m_peak[price_col]:.2f}"
            c3_sub = f"Low: ${m_low[price_col]:.2f} ({m_low['Month_Name']})"
        else:
            c1_sub = f"YoY: {sel_row['YoY_Diff_Fmt']} ({sel_row['YoY_Pct_Fmt']})"
            c3_label = f"vs. U.S. ({sel_yr_a})"
            c3_val = sel_row["Offset_US_Fmt"]
            c3_sub = f"U.S.: {sel_row['US_Price_Fmt']}"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"{sel_yr_a} Annual {central_word_title}",
            f"${sel_price:.2f}",
            c1_sub,
        )
        col2.metric(
            f"vs. {len(ann)}-Yr {central_word_title}",
            f"{diff_vs_central:+.2f}/gal",
            f"{pct_vs_central:+.1f}% vs. ${central_ann:.2f}",
        )
        col3.metric(c3_label, c3_val, c3_sub)
        col4.metric(
            f"Rank ({start_year}–{end_year})",
            f"#{rank_pos} of {len(ann)}",
            f"{sel_row['Pct_Start_Fmt']} vs. {start_year}",
        )

    ann_records = ann[
        [
            "Year_Str",
            "Price",
            "US_Price",
            "Price_Fmt",
            "US_Price_Fmt",
            "Offset_US_Fmt",
            "YoY_Diff_Fmt",
            "YoY_Pct_Fmt",
            "Pct_Start_Fmt",
            "Is_Selected",
        ]
    ].to_dict(orient="records")

    max_both_curves = float(max(ann["Price"].max(), ann["US_Price"].max()))
    y_max_a = float(np.ceil(max_both_curves * 2) / 2 + 0.25)
    locked_y_scale_a = alt.Scale(domain=[0, y_max_a], nice=False, clamp=True)
    all_year_strs = ann["Year_Str"].tolist()
    tick_step = max(3, int(math.ceil((end_year - start_year + 1) / 14)))
    tick_years = [str(y) for y in range(start_year, end_year + 1, tick_step)]

    shared_x_scale = alt.Scale(
        type="band",
        domain=all_year_strs,
        paddingInner=0.22,
        paddingOuter=0.35,
    )
    shared_x_axis = alt.Axis(
        values=tick_years,
        labelAngle=0,
        labelFontSize=12,
        titleFontSize=13,
        titlePadding=8,
        labelColor="#333333",
        titleColor="#111111",
        grid=True,
        gridColor="#EAEAEA",
    )

    base_annual = alt.Chart(alt.Data(values=ann_records)).encode(
        x=alt.X("Year_Str:O", title="Year", scale=shared_x_scale, axis=shared_x_axis)
    )

    area_layer = base_annual.mark_area(
        line={"color": "#25858e", "strokeWidth": 2.5},
        color="#25858e",
        opacity=0.10 if is_single_annual_year else 0.16,
    ).encode(
        y=alt.Y(
            "Price:Q",
            title=y_label_annual,
            scale=locked_y_scale_a,
            axis=alt.Axis(
                grid=True,
                gridColor="#EAEAEA",
                labelFontSize=12,
                titleFontSize=13,
                titlePadding=8,
                minExtent=38,
                labelColor="#333333",
                titleColor="#111111",
            ),
        )
    )

    ann_tooltips = [
        alt.Tooltip("Year_Str:N", title="Year"),
        alt.Tooltip("Price_Fmt:N", title=f"{geo_short} Annual {central_word_title}"),
    ]
    if not is_us_benchmark:
        ann_tooltips.extend([
            alt.Tooltip("US_Price_Fmt:N", title=f"U.S. Benchmark {central_word_title}"),
            alt.Tooltip("Offset_US_Fmt:N", title="Offset vs. U.S. ($)"),
        ])
    ann_tooltips.extend([
        alt.Tooltip("YoY_Diff_Fmt:N", title="YoY Change ($)"),
        alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
        alt.Tooltip("Pct_Start_Fmt:N", title=f"% Change vs. {start_year}"),
    ])

    points_layer = base_annual.mark_circle(size=70, color="#1f6f78").encode(
        y=alt.Y("Price:Q", scale=locked_y_scale_a),
        opacity=alt.condition("datum.Is_Selected", alt.value(0.95), alt.value(0.28)),
        tooltip=ann_tooltips,
    )

    central_rule = (
        alt.Chart(alt.Data(values=[{"central_val": central_ann}]))
        .mark_rule(color="#555555", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(y=alt.Y("central_val:Q", scale=locked_y_scale_a))
    )

    annual_layers = []

    if show_ci:
        ci_band_df = [{"ci_low": ci_low_ann, "ci_high": ci_high_ann}]
        ci_band_rect = (
            alt.Chart(alt.Data(values=ci_band_df))
            .mark_rect(color="#555555", opacity=0.11)
            .encode(y=alt.Y("ci_low:Q", scale=locked_y_scale_a), y2=alt.Y2("ci_high:Q"))
        )
        ci_upper_rule = (
            alt.Chart(alt.Data(values=[{"val": ci_high_ann}]))
            .mark_rule(color="#666666", strokeDash=[2, 3], strokeWidth=1.0, opacity=0.75)
            .encode(y=alt.Y("val:Q", scale=locked_y_scale_a))
        )
        ci_lower_rule = (
            alt.Chart(alt.Data(values=[{"val": ci_low_ann}]))
            .mark_rule(color="#666666", strokeDash=[2, 3], strokeWidth=1.0, opacity=0.75)
            .encode(y=alt.Y("val:Q", scale=locked_y_scale_a))
        )
        annual_layers.extend([ci_band_rect, ci_upper_rule, ci_lower_rule])

    annual_layers.extend([area_layer, points_layer, central_rule])

    # When a State or PADD region is chosen, overlay a subtle dotted U.S. National reference curve
    if not is_us_benchmark:
        us_ref_line = base_annual.mark_line(
            color="#888888",
            strokeDash=[4, 4],
            strokeWidth=1.6,
            opacity=0.85,
        ).encode(
            y=alt.Y("US_Price:Q", scale=locked_y_scale_a),
            tooltip=[
                alt.Tooltip("Year_Str:N", title="Year"),
                alt.Tooltip("US_Price_Fmt:N", title=f"U.S. Benchmark {central_word_title}"),
                alt.Tooltip("Price_Fmt:N", title=f"{geo_short} Annual {central_word_title}"),
                alt.Tooltip("Offset_US_Fmt:N", title="Offset vs. U.S. ($)"),
            ],
        )
        annual_layers.append(us_ref_line)

    if is_single_annual_year:
        sel_records = [r for r in ann_records if r["Is_Selected"]]
        for r in sel_records:
            r["Callout_Text"] = f"{r['Year_Str']}: {r['Price_Fmt']}"
        sel_data = alt.Data(values=sel_records)
        sel_vline = (
            alt.Chart(sel_data)
            .mark_rule(color="#111111", strokeDash=[3, 3], strokeWidth=1.5, opacity=0.8)
            .encode(x=alt.X("Year_Str:O", scale=shared_x_scale))
        )
        if is_us_benchmark:
            sel_halo = (
                alt.Chart(sel_data)
                .mark_circle(size=230, color="#d95f02", opacity=0.25)
                .encode(x=alt.X("Year_Str:O", scale=shared_x_scale), y=alt.Y("Price:Q", scale=locked_y_scale_a))
            )
            sel_dot = (
                alt.Chart(sel_data)
                .mark_circle(size=120, color="#d95f02", stroke="white", strokeWidth=1.6)
                .encode(
                    x=alt.X("Year_Str:O", scale=shared_x_scale),
                    y=alt.Y("Price:Q", scale=locked_y_scale_a),
                    tooltip=ann_tooltips,
                )
            )
            sel_label = (
                alt.Chart(sel_data)
                .mark_text(dy=-14, fontSize=11, fontWeight="bold", color="#111111")
                .encode(
                    x=alt.X("Year_Str:O", scale=shared_x_scale),
                    y=alt.Y("Price:Q", scale=locked_y_scale_a),
                    text="Callout_Text:N",
                )
            )
            annual_layers.extend([sel_vline, sel_halo, sel_dot, sel_label])
        else:
            sel_dot = (
                alt.Chart(sel_data)
                .mark_circle(size=120, color="#111111", stroke="white", strokeWidth=1.6)
                .encode(
                    x=alt.X("Year_Str:O", scale=shared_x_scale),
                    y=alt.Y("Price:Q", scale=locked_y_scale_a),
                    tooltip=ann_tooltips,
                )
            )
            annual_layers.extend([sel_vline, sel_dot])

    basis_tag = f"{latest_label} Dollars" if is_real else "Nominal"
    us_note = " · Dotted gray curve = U.S. national benchmark" if not is_us_benchmark else ""
    if is_single_annual_year:
        sel_r_sub = ann.loc[ann["Year"] == sel_yr_a].iloc[0]
        subtitle_a = (
            f"Highlighted: {sel_yr_a} ({sel_r_sub['Price_Fmt']}/gal, YoY {sel_r_sub['YoY_Pct_Fmt']}) · "
            f"Dashed line = {len(ann)}-yr {central_word_lower} (${central_ann:.2f}/gal, {basis_tag}){us_note}"
        )
    elif show_ci:
        subtitle_a = (
            f"Dashed line = {len(ann)}-year {central_word_lower} (${central_ann:.2f}/gal; "
            f"95% CI: {ci_ann_fmt}, {basis_tag}){us_note}"
        )
    else:
        subtitle_a = (
            f"Dashed line = {len(ann)}-year {central_word_lower} (${central_ann:.2f}/gal, {basis_tag}){us_note}"
        )

    if show_annual_events:
        annual_events_meta = [
            {
                "Year": 1998,
                "Category": "YoY Event 1: Macro & Pandemic Demand Shocks",
                "Above_Label": "Asian Crisis",
                "Below_Label": "",
                "Event": "Asian Financial Crisis slumps global oil demand (34-year price low)",
            },
            {
                "Year": 2000,
                "Category": "YoY Event 2: OPEC+ & U.S. Shale Supply",
                "Above_Label": "OPEC Cuts",
                "Below_Label": "",
                "Event": "Coordinated OPEC production cuts drive +30.6% annual price surge",
            },
            {
                "Year": 2005,
                "Category": "Monthly Event 3: Gulf Coast Hurricanes",
                "Above_Label": "Katrina & Rita",
                "Below_Label": "",
                "Event": "Hurricanes Katrina & Rita knock Gulf Coast offshore rigs and refineries offline",
            },
            {
                "Year": 2008,
                "Category": "YoY Event 1: Macro & Pandemic Demand Shocks",
                "Above_Label": "Financial Crisis",
                "Below_Label": "",
                "Event": "Global Financial Crisis spike in 2008 followed by -27.5% crash in 2009",
            },
            {
                "Year": 2011,
                "Category": "YoY Event 4: SPR Releases & Blending Policy",
                "Above_Label": "",
                "Below_Label": "SPR Release",
                "Event": "Emergency 30.6M-bbl Strategic Petroleum Reserve (SPR) drawdown & ethanol blending mandates",
            },
            {
                "Year": 2012,
                "Category": "YoY Event 3: Geopolitical Conflicts & Sanctions",
                "Above_Label": "Libya / Iran Sanctions",
                "Below_Label": "",
                "Event": "Arab Spring, Libyan civil war supply outage & Iran oil sanctions",
            },
            {
                "Year": 2015,
                "Category": "YoY Event 2: OPEC+ & U.S. Shale Supply",
                "Above_Label": "U.S. Shale Glut",
                "Below_Label": "",
                "Event": "U.S. shale oil boom & OPEC price war trigger -27.7% annual drop",
            },
            {
                "Year": 2020,
                "Category": "YoY Event 1: Macro & Pandemic Demand Shocks",
                "Above_Label": "",
                "Below_Label": "COVID-19",
                "Event": "COVID-19 lockdown drop (-16.7%) followed by 2021 reopening surge (+38.7%)",
            },
            {
                "Year": 2022,
                "Category": "YoY Event 3: Geopolitical Conflicts & Sanctions",
                "Above_Label": "Russia–Ukraine & SPR",
                "Below_Label": "",
                "Event": "Russia–Ukraine War supply rerouting cushioned by 180M-bbl SPR release",
            },
            {
                "Year": 2024,
                "Category": "YoY Event 2: OPEC+ & U.S. Shale Supply",
                "Above_Label": "Non-OPEC Supply",
                "Below_Label": "",
                "Event": "Record U.S. & Americas oil output eases annual prices across 2023–2025",
            },
            {
                "Year": 2026,
                "Category": "YoY Event 3: Geopolitical Conflicts & Sanctions",
                "Above_Label": "Iran War",
                "Below_Label": "",
                "Event": "U.S.–Israel war with Iran (begun Feb 28, 2026) & Strait of Hormuz crisis trigger sharp oil and pump price surge",
            },
        ]

        events_df = pd.DataFrame(annual_events_meta)
        events_merged = pd.merge(
            events_df,
            ann[["Year", "Year_Str", "Price", "Price_Fmt", "YoY_Diff_Fmt", "YoY_Pct_Fmt", "YoY_Pct"]],
            on="Year",
            how="inner",
        )
        events_merged["YoY_Pct_Plot"] = events_merged["YoY_Pct"].fillna(0.0)
        event_records = events_merged[
            [
                "Year_Str",
                "Category",
                "Above_Label",
                "Below_Label",
                "Event",
                "Price",
                "Price_Fmt",
                "YoY_Diff_Fmt",
                "YoY_Pct_Fmt",
                "YoY_Pct_Plot",
            ]
        ].to_dict(orient="records")

        event_categories = [
            "YoY Event 1: Macro & Pandemic Demand Shocks",
            "YoY Event 2: OPEC+ & U.S. Shale Supply",
            "YoY Event 3: Geopolitical Conflicts & Sanctions",
            "YoY Event 4: SPR Releases & Blending Policy",
            "Monthly Event 3: Gulf Coast Hurricanes",
        ]
        event_colors = ["#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

        event_rules = (
            alt.Chart(alt.Data(values=event_records))
            .mark_rule(strokeDash=[3, 3], strokeWidth=1.2, opacity=0.65)
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=event_categories, range=event_colors),
                    legend=None,
                ),
            )
        )

        event_diamonds = (
            alt.Chart(alt.Data(values=event_records))
            .mark_point(shape="diamond", size=140, filled=True, stroke="white", strokeWidth=1.2)
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                y=alt.Y("Price:Q", scale=locked_y_scale_a),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=event_categories, range=event_colors),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Year_Str:N", title="Year"),
                    alt.Tooltip("Category:N", title="Driver Category"),
                    alt.Tooltip("Event:N", title="Event"),
                    alt.Tooltip("Price_Fmt:N", title=f"{geo_short} Annual {central_word_title}"),
                    alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
                ],
            )
        )

        above_center_records = [r for r in event_records if r["Above_Label"] and r["Year_Str"] != str(end_year)]
        above_edge_records = [r for r in event_records if r["Above_Label"] and r["Year_Str"] == str(end_year)]

        above_texts = (
            alt.Chart(alt.Data(values=above_center_records))
            .mark_text(dy=-11, fontSize=9.8, fontWeight="bold", color="#111111")
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                y=alt.Y("Price:Q", scale=locked_y_scale_a),
                text="Above_Label:N",
            )
        )

        above_edge_texts = (
            alt.Chart(alt.Data(values=above_edge_records))
            .mark_text(dy=-11, dx=4, align="right", fontSize=9.8, fontWeight="bold", color="#111111")
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                y=alt.Y("Price:Q", scale=locked_y_scale_a),
                text="Above_Label:N",
            )
        )

        below_texts = (
            alt.Chart(alt.Data(values=[r for r in event_records if r["Below_Label"]]))
            .mark_text(dy=16, fontSize=9.8, fontWeight="bold", color="#111111")
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                y=alt.Y("Price:Q", scale=locked_y_scale_a),
                text="Below_Label:N",
            )
        )

        annual_layers.extend([event_rules, event_diamonds, above_texts, above_edge_texts, below_texts])

    if is_single_annual_year:
        annual_chart_title = (
            f"Annual {central_word_title} {geo_short} Regular Retail Gasoline Price — {sel_yr_a} ({start_year}–{end_year})"
            if not is_us_benchmark
            else f"Annual {central_word_title} U.S. Regular Retail Gasoline Price — {sel_yr_a} ({start_year}–{end_year})"
        )
    else:
        annual_chart_title = (
            f"Annual {central_word_title} {geo_short} Regular Retail Gasoline Price ({start_year}–{end_year})"
            if not is_us_benchmark
            else f"Annual {central_word_title} U.S. Regular Retail Gasoline Price ({start_year}–{end_year})"
        )

    annual_line_chart = (
        alt.layer(*annual_layers)
        .resolve_scale(color="independent", y="shared")
        .properties(
            width=640,
            height=370,
            title=alt.TitleParams(
                text=annual_chart_title,
                subtitle=subtitle_a,
                anchor="start",
                fontSize=16.5,
                subtitleFontSize=12.5,
                color="#111111",
                subtitleColor="#444444",
                offset=14,
            ),
        )
    )

    yoy_all = ann.copy()
    yoy_all["YoY_Pct_Plot"] = yoy_all["YoY_Pct"].fillna(0.0)
    yoy_records = yoy_all[
        ["Year_Str", "YoY_Pct_Plot", "Direction", "Price_Fmt", "YoY_Diff_Fmt", "YoY_Pct_Fmt", "Is_Selected"]
    ].to_dict(orient="records")

    yoy_bars = (
        alt.Chart(alt.Data(values=yoy_records))
        .mark_bar()
        .encode(
            x=alt.X("Year_Str:O", title="Year", scale=shared_x_scale, axis=shared_x_axis),
            y=alt.Y(
                "YoY_Pct_Plot:Q",
                title="Year-over-Year Change (%)",
                axis=alt.Axis(
                    grid=True,
                    gridColor="#EAEAEA",
                    labelFontSize=12,
                    titleFontSize=13,
                    titlePadding=8,
                    minExtent=38,
                    labelColor="#333333",
                    titleColor="#111111",
                ),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Increase", "Decrease"], range=["#25858e", "#d95f02"]),
                legend=None,
            ),
            opacity=alt.condition("datum.Is_Selected", alt.value(1.0), alt.value(0.28)),
            tooltip=[
                alt.Tooltip("Year_Str:N", title="Year"),
                alt.Tooltip("Price_Fmt:N", title=f"{geo_short} Annual {central_word_title}"),
                alt.Tooltip("YoY_Diff_Fmt:N", title="YoY Change ($)"),
                alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
            ],
        )
    )

    yoy_layers = [yoy_bars]
    if show_annual_events:
        yoy_event_rules = (
            alt.Chart(alt.Data(values=event_records))
            .mark_rule(strokeDash=[3, 3], strokeWidth=1.2, opacity=0.65)
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=event_categories, range=event_colors),
                    legend=None,
                ),
            )
        )
        yoy_event_diamonds = (
            alt.Chart(alt.Data(values=event_records))
            .mark_point(shape="diamond", size=115, filled=True, stroke="white", strokeWidth=1.1)
            .encode(
                x=alt.X("Year_Str:O", scale=shared_x_scale),
                y=alt.Y("YoY_Pct_Plot:Q"),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=event_categories, range=event_colors),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Year_Str:N", title="Year"),
                    alt.Tooltip("Category:N", title="Driver Category"),
                    alt.Tooltip("Event:N", title="Event"),
                    alt.Tooltip("Price_Fmt:N", title=f"{geo_short} Annual {central_word_title}"),
                    alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
                ],
            )
        )
        yoy_layers.extend([yoy_event_rules, yoy_event_diamonds])

    yoy_bar_chart = (
        alt.layer(*yoy_layers)
        .resolve_scale(color="independent")
        .properties(
            width=640,
            height=250,
            title=alt.TitleParams(
                text=f"{geo_short} Year-over-Year Percentage Price Change ({start_year}–{end_year})",
                subtitle=f"Annual percentage swing compared to the prior calendar year ({start_year} baseline has no prior year)",
                anchor="start",
                fontSize=16,
                subtitleFontSize=12.5,
                color="#111111",
                subtitleColor="#444444",
                offset=14,
            ),
        )
    )

    combined_annual_chart = (
        alt.vconcat(annual_line_chart, yoy_bar_chart, spacing=28)
        .resolve_scale(x="shared")
        .configure_view(stroke="#333333", strokeWidth=0.8, fill="white")
        .configure(background="white", padding={"top": 20, "bottom": 18, "left": 14, "right": 20})
    )

    st.altair_chart(combined_annual_chart, width="stretch")

    annual_table_dict = {
        "Year": ann["Year_Str"],
        f"{geo_short} {central_word_title} Price": ann["Price_Fmt"],
    }
    if not is_us_benchmark:
        annual_table_dict["U.S. Benchmark"] = ann["US_Price_Fmt"]
        annual_table_dict["Offset vs. U.S. ($)"] = ann["Offset_US_Fmt"]
    annual_table_dict.update({
        "YoY Change ($)": ann["YoY_Diff_Fmt"],
        "YoY Change (%)": ann["YoY_Pct_Fmt"],
        f"% Change vs. {start_year}": ann["Pct_Start_Fmt"],
    })
    display_annual = pd.DataFrame(annual_table_dict).iloc[::-1]
    if is_single_annual_year:
        display_annual = display_annual[display_annual["Year"] == str(sel_yr_a)]

    st.dataframe(display_annual, height="content", width="stretch", hide_index=True)

    ytd_yrs = sorted(ann.loc[ann.get("Is_YTD", pd.Series(False, index=ann.index)), "Year"].astype(int).tolist())
    ytd_note = (
        f" Note: {', '.join(map(str, ytd_yrs))} reflects year-to-date monthly averages pending EIA's annual survey release."
        if ytd_yrs
        else ""
    )
    st.caption(
        f"Sources: [U.S. EIA Retail Gasoline Prices ({start_year}–{end_year})](https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_a.htm), "
        f"[EIA SEDS State Price Estimates](https://www.eia.gov/state/seds/), & "
        f"[BLS CPI-U via FRED (CPIAUCSL)]({FRED_CPI_URL}).{ytd_note}"
    )

with st.expander("View Python Source Code"):
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
