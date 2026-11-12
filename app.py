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
    df.groupby("Month_Num")[price_col]
    .agg(Count="count", Mean="mean")
    .reindex(range(1, 13))
)
summary["Month"] = month_names

jan_mean = summary.loc[1, "Mean"]
summary["Change_vs_Jan"] = summary["Mean"] - jan_mean
summary["Pct_Change_vs_Jan"] = (summary["Change_vs_Jan"] / jan_mean) * 100

prev_mean = summary["Mean"].shift(1)
prev_mean.iloc[0] = summary["Mean"].iloc[-1]  # Wrap Dec -> Jan for January's comparison
summary["MoM_Pct"] = ((summary["Mean"] - prev_mean) / prev_mean) * 100

# Build chart
fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

cmap = plt.get_cmap("viridis")
colors = [cmap(i / 11.0) for i in range(12)]
x = np.arange(12)
ax.bar(x, summary["Mean"], width=0.82, color=colors, zorder=3)

y_max = np.ceil(summary["Mean"].max() * 2) / 2 + 0.25
ax.set_ylim(-0.08, max(2.85, y_max))
ax.set_xlim(-0.65, 11.65)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_xticks(x)
ax.set_xticklabels(month_names, fontsize=11, color="#333333")
ax.grid(True, which="major", axis="both", color="#E5E5E5", linewidth=0.9, zorder=0)
ax.grid(True, which="minor", axis="y", color="#F0F0F0", linewidth=0.7, zorder=0)
ax.set_xlabel("Month", fontsize=13, labelpad=8)
ax.set_ylabel(f"Average Price ({latest_label} $)" if is_real else "Average Price ($)", fontsize=13, labelpad=8)

date_span = f"{df['Date'].min():%B %Y} - {df['Date'].max():%B %Y}"
subtitle = f"{date_span} (Inflation-Adjusted to {latest_label} Dollars)" if is_real else date_span
fig.text(0.075, 0.945, "Average Price of Unleaded Regular Gasoline by Month of the Year", fontsize=15.5, ha="left")
fig.text(0.075, 0.905, subtitle, fontsize=12.5, ha="left")

legend_handles = [mpatches.Patch(color=colors[i], label=month_names[i]) for i in range(12)]
ax.legend(
    handles=legend_handles,
    title="month",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=10.5,
    title_fontsize=12.5,
)
plt.subplots_adjust(left=0.075, right=0.88, top=0.87, bottom=0.09)

st.pyplot(fig)

# Display full 12-month table without a vertical scrollbar
display_df = pd.DataFrame({
    "Month": summary["Month"],
    "Observations": summary["Count"],
    "Average Price": summary["Mean"].map("${:.2f}".format),
    "Change vs. January": summary["Change_vs_Jan"].map("{:+.2f}".format),
    "% Change vs. January": summary["Pct_Change_vs_Jan"].map("{:+.1f}%".format),
    "% Change vs. Prev. Month": summary["MoM_Pct"].map("{:+.1f}%".format),
})
st.dataframe(display_df, height="content", width="stretch", hide_index=True)

st.caption(
    f"Sources: U.S. EIA (EMM_EPMR_PTE_NUS_DPG) & BLS CPI-U via FRED (CPIAUCSL), "
    f"{date_span} ({len(df)} monthly observations)."
)

with st.expander("View Python Source Code"):
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
