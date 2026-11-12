import io
import requests
import pandas as pd
import numpy as np
import altair as alt
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
viridis_hex = [
    "#440154", "#482173", "#433e85", "#38588c", "#2d708e", "#25858e",
    "#1e9b8a", "#2ab07f", "#52c569", "#86d549", "#c2df23", "#fde725",
]

df["Month_Num"] = df["Date"].dt.month
summary = df.groupby("Month_Num")[price_col].agg(Count="count", Mean="mean").reindex(range(1, 13))
summary["Month"] = month_names

jan_mean = summary.loc[1, "Mean"]
summary["Change_vs_Jan"] = summary["Mean"] - jan_mean
summary["Pct_Change_vs_Jan"] = (summary["Change_vs_Jan"] / jan_mean) * 100

prev_mean = summary["Mean"].shift(1)
prev_mean.iloc[0] = summary["Mean"].iloc[-1]  # Wrap Dec -> Jan for January's comparison
summary["MoM_Pct"] = ((summary["Mean"] - prev_mean) / prev_mean) * 100

# Pre-format strings so hover tooltips and table stay 100% consistent
summary["Avg_Price_Fmt"] = summary["Mean"].map("${:.2f}".format)
summary["Change_Jan_Fmt"] = summary["Change_vs_Jan"].map("{:+.2f}".format)
summary["Pct_Jan_Fmt"] = summary["Pct_Change_vs_Jan"].map("{:+.1f}%".format)
summary["MoM_Pct_Fmt"] = summary["MoM_Pct"].map("{:+.1f}%".format)

y_max = float(max(2.85, np.ceil(summary["Mean"].max() * 2) / 2 + 0.25))
y_grid_vals = [round(float(v), 1) for v in np.arange(0, y_max + 0.01, 0.5)]
date_span = f"{df['Date'].min():%B %Y} - {df['Date'].max():%B %Y}"
subtitle = f"{date_span} (Inflation-Adjusted to {latest_label} Dollars)" if is_real else date_span
y_label = f"Average Price ({latest_label} $)" if is_real else "Average Price ($)"

chart_records = summary.reset_index(drop=True).to_dict(orient="records")

chart = (
    alt.Chart(alt.Data(values=chart_records))
    .mark_bar()
    .encode(
        x=alt.X(
            "Month:N",
            sort=month_names,
            title="Month",
            scale=alt.Scale(paddingInner=0.16, paddingOuter=0.12),
            axis=alt.Axis(
                labelAngle=0,
                labelFontSize=12,
                titleFontSize=14,
                labelColor="#333333",
                titleColor="#111111",
                grid=True,
                gridColor="#EAEAEA",
            ),
        ),
        y=alt.Y(
            "Mean:Q",
            title=y_label,
            scale=alt.Scale(domain=[0, y_max]),
            axis=alt.Axis(
                values=y_grid_vals,
                labelExpr="datum.value % 1 === 0 ? datum.value : ''",
                grid=True,
                gridColor="#EAEAEA",
                labelFontSize=12,
                titleFontSize=14,
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
        tooltip=[
            alt.Tooltip("Avg_Price_Fmt:N", title="Average Price"),
            alt.Tooltip("Change_Jan_Fmt:N", title="Change vs. January"),
            alt.Tooltip("Pct_Jan_Fmt:N", title="% Change vs. January"),
            alt.Tooltip("MoM_Pct_Fmt:N", title="% Change vs. Prev. Month"),
            alt.Tooltip("Count:Q", title="Observations"),
        ],
    )
    .properties(
        height=480,
        title=alt.TitleParams(
            text="Average Price of Unleaded Regular Gasoline by Month of the Year",
            subtitle=subtitle,
            anchor="start",
            fontSize=18,
            subtitleFontSize=14,
            color="#111111",
            subtitleColor="#333333",
            offset=12,
        ),
    )
    .configure_view(stroke="#333333", strokeWidth=0.8, fill="white")
    .configure(background="white")
)

st.altair_chart(chart, width="stretch")

# Display full 12-month table without a vertical scrollbar
display_df = pd.DataFrame({
    "Month": summary["Month"],
    "Observations": summary["Count"],
    "Average Price": summary["Avg_Price_Fmt"],
    "Change vs. January": summary["Change_Jan_Fmt"],
    "% Change vs. January": summary["Pct_Jan_Fmt"],
    "% Change vs. Prev. Month": summary["MoM_Pct_Fmt"],
})
st.dataframe(display_df, height="content", width="stretch", hide_index=True)

st.caption(
    f"Sources: U.S. EIA (EMM_EPMR_PTE_NUS_DPG) & BLS CPI-U via FRED (CPIAUCSL), "
    f"{date_span} ({len(df)} monthly observations)."
)

with st.expander("View Python Source Code"):
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
