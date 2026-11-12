import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import streamlit as st

st.set_page_config(page_title="U.S. Regular Gasoline Seasonal Trends", layout="centered")

EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGm.xls"
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


@st.cache_data(ttl=86400)  # Refresh cached data once every 24 hours
def load_merged_data() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. Load EIA monthly regular retail gasoline prices
    resp = requests.get(EIA_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    gas = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data 1", skiprows=2)
    gas.columns = ["Date", "Nominal_Price"]
    gas["Date"] = pd.to_datetime(gas["Date"])
    gas["Nominal_Price"] = pd.to_numeric(gas["Nominal_Price"], errors="coerce")
    gas = gas.dropna(subset=["Date", "Nominal_Price"]).copy()
    gas["YearMonth"] = gas["Date"].dt.to_period("M")

    # 2. Load BLS CPI-U from FRED and interpolate any single-month gap
    cpi = pd.read_csv(FRED_CPI_URL, na_values=".")
    cpi.columns = ["Date", "CPI"]
    cpi["Date"] = pd.to_datetime(cpi["Date"])
    cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="coerce").interpolate(method="linear").ffill()
    cpi["YearMonth"] = cpi["Date"].dt.to_period("M")

    # 3. Merge and compute constant dollars anchored to the latest available CPI month
    merged = pd.merge(gas, cpi[["YearMonth", "CPI"]], on="YearMonth", how="left")
    merged["CPI"] = merged["CPI"].ffill()
    latest_cpi = merged["CPI"].iloc[-1]
    merged["Real_Price"] = merged["Nominal_Price"] * (latest_cpi / merged["CPI"])

    return merged.sort_values("Date").reset_index(drop=True)


df = load_merged_data()
latest_label = f"{df['Date'].max():%B %Y}"

mode = st.radio(
    "Price Basis:",
    options=["Nominal Dollars (Unadjusted)", f"Inflation-Adjusted ({latest_label} Dollars)"],
    horizontal=True,
)
is_real = mode.startswith("Inflation-Adjusted")
price_col = "Real_Price" if is_real else "Nominal_Price"

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df["Month_Num"] = df["Date"].dt.month

summary = (
    df.groupby
