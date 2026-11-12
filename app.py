import io
import requests
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

st.set_page_config(page_title="U.S. Regular Gasoline Price Trends", layout="centered")

EIA_MONTHLY_URL = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGm.xls"
EIA_ANNUAL_URL = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGa.xls"
LOCAL_ANNUAL_CSV = "U.S._Regular_All_Formulations_Retail_Gasoline_Prices.csv"
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


@st.cache_data(ttl=86400)  # Refresh cached data once every 24 hours
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. Load monthly BLS CPI-U from FRED and interpolate any single-month gap
    cpi = pd.read_csv(FRED_CPI_URL, na_values=".")
    cpi.columns = ["Date", "CPI"]
    cpi["Date"] = pd.to_datetime(cpi["Date"])
    cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="coerce").interpolate(method="linear").ffill()
    cpi["YearMonth"] = cpi["Date"].dt.to_period("M")
    cpi["Year"] = cpi["Date"].dt.year

    # 2. Load EIA monthly regular retail gasoline prices
    resp_m = requests.get(EIA_MONTHLY_URL, headers=headers, timeout=20)
    resp_m.raise_for_status()
    monthly = pd.read_excel(io.BytesIO(resp_m.content), sheet_name="Data 1", skiprows=2)
    monthly.columns = ["Date", "Nominal_Price"]
    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly["Nominal_Price"] = pd.to_numeric(monthly["Nominal_Price"], errors="coerce")
    monthly = monthly.dropna(subset=["Date", "Nominal_Price"]).copy()
    monthly["YearMonth"] = monthly["Date"].dt.to_period("M")

    monthly = pd.merge(monthly, cpi[["YearMonth", "CPI"]], on="YearMonth", how="left")
    monthly["CPI"] = monthly["CPI"].ffill()
    latest_cpi = float(monthly["CPI"].iloc[-1])
    monthly["Real_Price"] = monthly["Nominal_Price"] * (latest_cpi / monthly["CPI"])
    monthly = monthly.sort_values("Date").reset_index(drop=True)

    # 3. Load EIA annual regular retail gasoline prices (live XLS with local CSV fallback)
    try:
        resp_a = requests.get(EIA_ANNUAL_URL, headers=headers, timeout=20)
        resp_a.raise_for_status()
        annual = pd.read_excel(io.BytesIO(resp_a.content), sheet_name="Data 1", skiprows=2)
        annual.columns = ["Date", "Nominal_Price"]
        annual["Year"] = pd.to_datetime(annual["Date"]).dt.year
    except Exception:
        annual = pd.read_csv(LOCAL_ANNUAL_CSV, skiprows=4)
        annual.columns = ["Year", "Nominal_Price"]
        annual["Year"] = pd.to_numeric(annual["Year"], errors="coerce").astype(int)

    annual["Nominal_Price"] = pd.to_numeric(annual["Nominal_Price"], errors="coerce")
    annual = annual[["Year", "Nominal_Price"]].dropna().sort_values("Year").reset_index(drop=True)

    # Compute annual average CPI to adjust annual gasoline prices to the same latest-CPI anchor
    annual_cpi = cpi.groupby("Year", as_index=False)["CPI"].mean()
    annual = pd.merge(annual, annual_cpi, on="Year", how="left")
    annual["CPI"] = annual["CPI"].ffill()
    annual["Real_Price"] = annual["Nominal_Price"] * (latest_cpi / annual["CPI"])

    return monthly, annual


monthly_df, annual_df = load_all_data()
latest_label = f"{monthly_df['Date'].max():%B %Y}"

mode = st.radio(
    "Price Basis:",
    options=["Nominal Dollars (Unadjusted)", f"Inflation-Adjusted ({latest_label} Dollars)"],
    horizontal=True,
)
is_real = mode.startswith("Inflation-Adjusted")
price_col = "Real_Price" if is_real else "Nominal_Price"
y_label = f"Average Price ({latest_label} $)" if is_real else "Average Price ($)"

tab_seasonal, tab_annual = st.tabs([
    "Seasonal Trends (By Month)",
    f"Annual Trends ({int(annual_df['Year'].min())}–{int(annual_df['Year'].max())})",
])

# ==============================================================================
# TAB 1: SEASONAL TRENDS BY CALENDAR MONTH
# ==============================================================================
with tab_seasonal:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    viridis_hex = [
        "#440154", "#482173", "#433e85", "#38588c", "#2d708e", "#25858e",
        "#1e9b8a", "#2ab07f", "#52c569", "#86d549", "#c2df23", "#fde725",
    ]

    monthly_df["Month_Num"] = monthly_df["Date"].dt.month
    summary = monthly_df.groupby("Month_Num")[price_col].agg(Count="count", Mean="mean").reindex(range(1, 13))
    summary["Month"] = month_names

    jan_mean = summary.loc[1, "Mean"]
    summary["Change_vs_Jan"] = summary["Mean"] - jan_mean
    summary["Pct_Change_vs_Jan"] = (summary["Change_vs_Jan"] / jan_mean) * 100

    prev_mean = summary["Mean"].shift(1)
    prev_mean.iloc[0] = summary["Mean"].iloc[-1]  # Wrap Dec -> Jan for January's comparison
    summary["MoM_Pct"] = ((summary["Mean"] - prev_mean) / prev_mean) * 100

    summary["Avg_Price_Fmt"] = summary["Mean"].map("${:.2f}".format)
    summary["Change_Jan_Fmt"] = summary["Change_vs_Jan"].map("{:+.2f}".format)
    summary["Pct_Jan_Fmt"] = summary["Pct_Change_vs_Jan"].map("{:+.1f}%".format)
    summary["MoM_Pct_Fmt"] = summary["MoM_Pct"].map("{:+.1f}%".format)

    y_max_m = float(max(2.85, np.ceil(summary["Mean"].max() * 2) / 2 + 0.25))
    y_grid_vals = [round(float(v), 1) for v in np.arange(0, y_max_m + 0.01, 0.5)]
    date_span = f"{monthly_df['Date'].min():%B %Y} - {monthly_df['Date'].max():%B %Y}"
    subtitle_m = f"{date_span} (Inflation-Adjusted to {latest_label} Dollars)" if is_real else date_span

    seasonal_chart = (
        alt.Chart(alt.Data(values=summary.reset_index(drop=True).to_dict(orient="records")))
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
                scale=alt.Scale(domain=[0, y_max_m]),
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
            height=460,
            title=alt.TitleParams(
                text="Average Price of Unleaded Regular Gasoline by Month of the Year",
                subtitle=subtitle_m,
                anchor="start",
                fontSize=17.5,
                subtitleFontSize=13.5,
                color="#111111",
                subtitleColor="#333333",
                offset=12,
            ),
        )
        .configure_view(stroke="#333333", strokeWidth=0.8, fill="white")
        .configure(background="white")
    )

    st.altair_chart(seasonal_chart, width="stretch")

    display_seasonal = pd.DataFrame({
        "Month": summary["Month"],
        "Observations": summary["Count"],
        "Average Price": summary["Avg_Price_Fmt"],
        "Change vs. January": summary["Change_Jan_Fmt"],
        "% Change vs. January": summary["Pct_Jan_Fmt"],
        "% Change vs. Prev. Month": summary["MoM_Pct_Fmt"],
    })
    st.dataframe(display_seasonal, height="content", width="stretch", hide_index=True)

    st.caption(
        f"Sources: U.S. EIA (EMM_EPMR_PTE_NUS_DPG) & BLS CPI-U via FRED (CPIAUCSL), "
        f"{date_span} ({len(monthly_df)} monthly observations)."
    )

# ==============================================================================
# TAB 2: ANNUAL TRENDS (1992–PRESENT)
# ==============================================================================
with tab_annual:
    ann = annual_df.copy()
    ann["Year_Str"] = ann["Year"].astype(str)
    ann["Price"] = ann[price_col]
    ann["YoY_Diff"] = ann["Price"].diff()
    ann["YoY_Pct"] = ann["Price"].pct_change() * 100

    start_year = int(ann["Year"].iloc[0])
    end_year = int(ann["Year"].iloc[-1])
    base_price = float(ann["Price"].iloc[0])
    end_price = float(ann["Price"].iloc[-1])
    ann["Pct_vs_Start"] = ((ann["Price"] - base_price) / base_price) * 100

    ann["Price_Fmt"] = ann["Price"].map("${:.2f}".format)
    ann["YoY_Diff_Fmt"] = ann["YoY_Diff"].map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}")
    ann["YoY_Pct_Fmt"] = ann["YoY_Pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%")
    ann["Pct_Start_Fmt"] = ann["Pct_vs_Start"].map("{:+.1f}%".format)
    ann["Direction"] = np.where(ann["YoY_Pct"].fillna(0) >= 0, "Increase", "Decrease")

    mean_ann = float(ann["Price"].mean())
    peak_row = ann.loc[ann["Price"].idxmax()]
    low_row = ann.loc[ann["Price"].idxmin()]
    cagr = ((end_price / base_price) ** (1.0 / (end_year - start_year)) - 1.0) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{len(ann)}-Year Average", f"${mean_ann:.2f}", f"{start_year}–{end_year} mean")
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

    ann_records = ann.to_dict(orient="records")
    y_max_a = float(np.ceil(ann["Price"].max() * 2) / 2 + 0.25)
    tick_years = [str(y) for y in range(start_year, end_year + 1, 3)]

    base_annual = alt.Chart(alt.Data(values=ann_records)).encode(
        x=alt.X(
            "Year_Str:O",
            title="Year",
            axis=alt.Axis(
                values=tick_years,
                labelAngle=0,
                labelFontSize=12,
                titleFontSize=14,
                labelColor="#333333",
                titleColor="#111111",
                grid=True,
                gridColor="#EAEAEA",
            ),
        )
    )

    area_layer = base_annual.mark_area(
        line={"color": "#25858e", "strokeWidth": 2.5},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(37, 133, 142, 0.28)", offset=0),
                alt.GradientStop(color="rgba(37, 133, 142, 0.02)", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        y=alt.Y(
            "Price:Q",
            title=y_label,
            scale=alt.Scale(domain=[0, y_max_a]),
            axis=alt.Axis(
                grid=True,
                gridColor="#EAEAEA",
                labelFontSize=12,
                titleFontSize=14,
                labelColor="#333333",
                titleColor="#111111",
            ),
        )
    )

    points_layer = base_annual.mark_circle(size=75, color="#1f6f78", opacity=0.95).encode(
        y=alt.Y("Price:Q"),
        tooltip=[
            alt.Tooltip("Year_Str:N", title="Year"),
            alt.Tooltip("Price_Fmt:N", title="Average Price"),
            alt.Tooltip("YoY_Diff_Fmt:N", title="YoY Change ($)"),
            alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
            alt.Tooltip("Pct_Start_Fmt:N", title=f"% Change vs. {start_year}"),
        ],
    )

    mean_rule = (
        alt.Chart(alt.Data(values=[{"mean_val": mean_ann}]))
        .mark_rule(color="#666666", strokeDash=[6, 4], strokeWidth=1.4)
        .encode(y="mean_val:Q")
    )

    subtitle_a = (
        f"Dashed line = {len(ann)}-year average (${mean_ann:.2f}/gal, {latest_label} Dollars) · Hover any year for details"
        if is_real
        else f"Dashed line = {len(ann)}-year average (${mean_ann:.2f}/gal, Nominal) · Hover any year for details"
    )

    annual_line_chart = (
        alt.layer(area_layer, points_layer, mean_rule)
        .properties(
            height=380,
            title=alt.TitleParams(
                text=f"Annual Average U.S. Regular Retail Gasoline Price ({start_year}–{end_year})",
                subtitle=subtitle_a,
                anchor="start",
                fontSize=17,
                subtitleFontSize=13,
                color="#111111",
                subtitleColor="#444444",
                offset=12,
            ),
        )
        .configure_view(stroke="#333333", strokeWidth=0.8, fill="white")
        .configure(background="white")
    )

    st.altair_chart(annual_line_chart, width="stretch")

    yoy_records = ann.dropna(subset=["YoY_Pct"]).to_dict(orient="records")
    yoy_tick_years = [str(y) for y in range(start_year + 1, end_year + 1, 3)]

    yoy_bar_chart = (
        alt.Chart(alt.Data(values=yoy_records))
        .mark_bar()
        .encode(
            x=alt.X(
                "Year_Str:O",
                title="Year",
                scale=alt.Scale(paddingInner=0.22),
                axis=alt.Axis(
                    values=yoy_tick_years,
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
                "YoY_Pct:Q",
                title="Year-over-Year Change (%)",
                axis=alt.Axis(
                    grid=True,
                    gridColor="#EAEAEA",
                    labelFontSize=12,
                    titleFontSize=14,
                    labelColor="#333333",
                    titleColor="#111111",
                ),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Increase", "Decrease"], range=["#25858e", "#d95f02"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Year_Str:N", title="Year"),
                alt.Tooltip("Price_Fmt:N", title="Average Price"),
                alt.Tooltip("YoY_Diff_Fmt:N", title="YoY Change ($)"),
                alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
            ],
        )
        .properties(
            height=250,
            title=alt.TitleParams(
                text=f"Year-over-Year Percentage Price Change ({start_year + 1}–{end_year})",
                subtitle="Annual percentage swing compared to the prior calendar year",
                anchor="start",
                fontSize=16,
                subtitleFontSize=12.5,
                color="#111111",
                subtitleColor="#444444",
                offset=10,
            ),
        )
        .configure_view(stroke="#333333", strokeWidth=0.8, fill="white")
        .configure(background="white")
    )

    st.altair_chart(yoy_bar_chart, width="stretch")

    display_annual = pd.DataFrame({
        "Year": ann["Year_Str"],
        "Average Price": ann["Price_Fmt"],
        "YoY Change ($)": ann["YoY_Diff_Fmt"],
        "YoY Change (%)": ann["YoY_Pct_Fmt"],
        f"% Change vs. {start_year}": ann["Pct_Start_Fmt"],
    }).iloc[::-1]  # Most recent year first

    st.dataframe(display_annual, height="content", width="stretch", hide_index=True)

    st.caption(
        f"Sources: U.S. EIA Annual Retail Gasoline Prices (EMM_EPMR_PTE_NUS_DPG, {start_year}–{end_year}) "
        f"& BLS CPI-U via FRED (CPIAUCSL)."
    )

with st.expander("View Python Source Code"):
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
