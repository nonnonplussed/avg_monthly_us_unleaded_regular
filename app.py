import io
import math
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

ctrl_col1, ctrl_col2 = st.columns([2, 1])
with ctrl_col1:
    mode = st.radio(
        "Price Basis:",
        options=["Nominal Dollars (Unadjusted)", f"Inflation-Adjusted ({latest_label} Dollars)"],
        horizontal=True,
    )
with ctrl_col2:
    stat_mode = st.radio(
        "Central Tendency:",
        options=["Mean", "Median"],
        horizontal=True,
    )

is_real = mode.startswith("Inflation-Adjusted")
is_median = stat_mode == "Median"
price_col = "Real_Price" if is_real else "Nominal_Price"
agg_func = "median" if is_median else "mean"
ci_func = median_ci_95 if is_median else mean_ci_95
stat_noun = "Median" if is_median else "Mean"
y_label_seasonal = f"{stat_noun} Price ({latest_label} $)" if is_real else f"{stat_noun} Price ($)"
y_label_annual = f"Mean Price ({latest_label} $)" if is_real else "Mean Price ($)"

# Category / overlay toggles placed directly below Price Basis & Central Tendency
ov_col1, ov_col2, ov_col3 = st.columns(3)
with ov_col1:
    show_seasonal_events = st.toggle(
        "Overlay Seasonal Events",
        value=False,
        help="Blend transitions (Mar, Nov) & summer driving season (May–Aug)",
    )
with ov_col2:
    show_annual_events = st.toggle(
        "Overlay Annual Events",
        value=False,
        help="Hurricanes, macro shocks, OPEC/shale, geopolitics & SPR releases",
    )
with ov_col3:
    show_ci = st.toggle(
        "Show 95% Confidence Interval",
        value=False,
        help="Adds the 95% confidence interval for the selected central tendency (Mean or Median)",
    )

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
    summary = monthly_df.groupby("Month_Num")[price_col].agg(Count="count", Value=agg_func).reindex(range(1, 13))
    summary["Month"] = month_names

    # Compute 95% CI bounds for each calendar month for both Mean and Median (to lock y-axis scale)
    ci_lows, ci_highs = [], []
    max_ci_both = 0.0
    for m_idx in range(1, 13):
        m_vals = monthly_df.loc[monthly_df["Month_Num"] == m_idx, price_col].to_numpy()
        lo_mean, hi_mean = mean_ci_95(m_vals)
        lo_med, hi_med = median_ci_95(m_vals)
        max_ci_both = max(max_ci_both, hi_mean, hi_med)
        if is_median:
            ci_lows.append(lo_med)
            ci_highs.append(hi_med)
        else:
            ci_lows.append(lo_mean)
            ci_highs.append(hi_mean)

    summary["CI_Low"] = ci_lows
    summary["CI_High"] = ci_highs
    summary["CI_Fmt"] = [f"${lo:.2f} – ${hi:.2f}" for lo, hi in zip(ci_lows, ci_highs)]

    jan_val = summary.loc[1, "Value"]
    summary["Change_vs_Jan"] = summary["Value"] - jan_val
    summary["Pct_Change_vs_Jan"] = (summary["Change_vs_Jan"] / jan_val) * 100

    prev_val = summary["Value"].shift(1)
    prev_val.iloc[0] = summary["Value"].iloc[-1]  # Wrap Dec -> Jan for January's comparison
    summary["MoM_Pct"] = ((summary["Value"] - prev_val) / prev_val) * 100

    summary["Val_Price_Fmt"] = summary["Value"].map("${:.2f}".format)
    summary["Change_Jan_Fmt"] = summary["Change_vs_Jan"].map("{:+.2f}".format)
    summary["Pct_Jan_Fmt"] = summary["Pct_Change_vs_Jan"].map("{:+.1f}%".format)
    summary["MoM_Pct_Fmt"] = summary["MoM_Pct"].map("{:+.1f}%".format)

    # Lock y-axis domain across Mean/Median, 95% CI On/Off, and Overlay On/Off so bar scale never shifts
    y_max_m = float(max(3.25, np.ceil(max_ci_both * 2) / 2 + 0.55))
    y_grid_vals = [round(float(v), 1) for v in np.arange(0, y_max_m + 0.01, 0.5)]
    locked_y_scale_m = alt.Scale(domain=[0, y_max_m], nice=False, clamp=True)

    date_span = f"{monthly_df['Date'].min():%B %Y} - {monthly_df['Date'].max():%B %Y}"
    subtitle_parts_m = []
    if is_real:
        subtitle_parts_m.append(f"{date_span} (Inflation-Adjusted to {latest_label} Dollars)")
    else:
        subtitle_parts_m.append(date_span)
    if show_ci:
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
            "Change_Jan_Fmt",
            "Pct_Jan_Fmt",
            "MoM_Pct_Fmt",
        ]
    ].to_dict(orient="records")

    seasonal_tooltips = [
        alt.Tooltip("Val_Price_Fmt:N", title=f"{stat_noun} Price"),
    ]
    if show_ci:
        seasonal_tooltips.append(alt.Tooltip("CI_Fmt:N", title=f"95% CI ({stat_noun})"))
    seasonal_tooltips.extend([
        alt.Tooltip("Change_Jan_Fmt:N", title="Change vs. January"),
        alt.Tooltip("Pct_Jan_Fmt:N", title="% Change vs. January"),
        alt.Tooltip("MoM_Pct_Fmt:N", title="% Change vs. Prev. Month"),
        alt.Tooltip("Count:Q", title="Observations"),
    ])

    seasonal_base = (
        alt.Chart(alt.Data(values=seasonal_records))
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

    if show_ci:
        ci_data = alt.Data(values=seasonal_records)
        ci_stem_halo = (
            alt.Chart(ci_data)
            .mark_rule(color="white", strokeWidth=3.6, opacity=0.85)
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("CI_Low:Q", scale=locked_y_scale_m),
                y2=alt.Y2("CI_High:Q"),
            )
        )
        ci_stem = (
            alt.Chart(ci_data)
            .mark_rule(color="#111111", strokeWidth=1.8)
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("CI_Low:Q", scale=locked_y_scale_m),
                y2=alt.Y2("CI_High:Q"),
                tooltip=[
                    alt.Tooltip("Month:N", title="Month"),
                    alt.Tooltip("Val_Price_Fmt:N", title=f"{stat_noun} Price"),
                    alt.Tooltip("CI_Fmt:N", title=f"95% CI ({stat_noun})"),
                ],
            )
        )
        ci_cap_top_halo = (
            alt.Chart(ci_data)
            .mark_tick(color="white", thickness=3.6, size=14, opacity=0.85)
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("CI_High:Q", scale=locked_y_scale_m),
            )
        )
        ci_cap_top = (
            alt.Chart(ci_data)
            .mark_tick(color="#111111", thickness=1.8, size=12)
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("CI_High:Q", scale=locked_y_scale_m),
            )
        )
        ci_cap_bot_halo = (
            alt.Chart(ci_data)
            .mark_tick(color="white", thickness=3.6, size=14, opacity=0.85)
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("CI_Low:Q", scale=locked_y_scale_m),
            )
        )
        ci_cap_bot = (
            alt.Chart(ci_data)
            .mark_tick(color="#111111", thickness=1.8, size=12)
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("CI_Low:Q", scale=locked_y_scale_m),
            )
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
        top_ref_col = "CI_High" if show_ci else "Value"
        top_lookup = dict(zip(summary["Month"], summary[top_ref_col]))
        max_ci_current = float(summary["CI_High"].max())
        headroom = y_max_m - max_ci_current

        # Position overlays high in the white space above the upper CI whiskers so there is zero collision
        y_summer_band = max_ci_current + headroom * 0.64
        y_blend_marker = max_ci_current + headroom * 0.30

        seasonal_event_points = [
            {
                "Month": "Mar",
                "anchor_top": float(top_lookup["Mar"]),
                "y_pos": y_blend_marker,
                "Event_Type": "Monthly Event 1: Fuel Blend Transitions & Maintenance",
                "Tag": "Summer-Blend Switch",
                "Detail": "EPA summer-blend transition & spring refinery maintenance (sharpest monthly rise)",
            },
            {
                "Month": "Nov",
                "anchor_top": float(top_lookup["Nov"]),
                "y_pos": y_blend_marker,
                "Event_Type": "Monthly Event 1: Fuel Blend Transitions & Maintenance",
                "Tag": "Winter-Blend Switch",
                "Detail": "Autumn switch back to lower-cost winter-blend gasoline (sharpest monthly drop)",
            },
            {
                "Month": "May",
                "anchor_top": float(top_lookup["May"]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Summer driving season begins around Memorial Day",
            },
            {
                "Month": "Jun",
                "anchor_top": float(top_lookup["Jun"]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Peak summer highway travel demand & summer-grade fuel plateau",
            },
            {
                "Month": "Jul",
                "anchor_top": float(top_lookup["Jul"]),
                "y_pos": y_summer_band,
                "Event_Type": "Monthly Event 2: Summer Driving Season Demand",
                "Tag": "",
                "Detail": "Mid-summer travel demand keeps refinery utilization elevated",
            },
            {
                "Month": "Aug",
                "anchor_top": float(top_lookup["Aug"]),
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
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("anchor_top:Q", scale=locked_y_scale_m),
                y2=alt.Y2("y_pos:Q"),
            )
        )

        span_records = [
            {"Month": m, "span_y": y_summer_band, "Detail": "Monthly Event 2: Summer Driving Season Demand (May–Aug)"}
            for m in ["May", "Jun", "Jul", "Aug"]
        ]

        span_line = (
            alt.Chart(alt.Data(values=span_records))
            .mark_line(color="#d95f02", strokeWidth=2.3, strokeDash=[5, 3])
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("span_y:Q", scale=locked_y_scale_m),
                tooltip=[alt.Tooltip("Detail:N", title="Seasonal Driver")],
            )
        )

        span_label = (
            alt.Chart(alt.Data(values=[{"Month": "Jun", "span_y": y_summer_band, "Text": "Summer Driving Season (May–Aug)"}]))
            .mark_text(dy=-11, dx=22, fontSize=10.5, fontWeight="bold", color="#111111")
            .encode(
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("span_y:Q", scale=locked_y_scale_m),
                text="Text:N",
            )
        )

        event_markers = (
            alt.Chart(alt.Data(values=seasonal_event_points))
            .mark_point(shape="diamond", size=135, filled=True, stroke="white", strokeWidth=1.2)
            .encode(
                x=alt.X("Month:N", sort=month_names),
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
                x=alt.X("Month:N", sort=month_names),
                y=alt.Y("y_pos:Q", scale=locked_y_scale_m),
                text="Tag:N",
            )
        )

        seasonal_layers.extend([leader_lines, span_line, span_label, event_markers, event_labels])

    seasonal_chart = (
        alt.layer(*seasonal_layers)
        .resolve_scale(color="independent", y="shared")
        .properties(
            height=460,
            title=alt.TitleParams(
                text=f"{stat_noun} Price of Unleaded Regular Gasoline by Month of the Year",
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
        f"{stat_noun} Price": summary["Val_Price_Fmt"],
    }
    if show_ci:
        seasonal_table_dict[f"95% CI ({stat_noun})"] = summary["CI_Fmt"]
    seasonal_table_dict.update({
        "Change vs. January": summary["Change_Jan_Fmt"],
        "% Change vs. January": summary["Pct_Jan_Fmt"],
        "% Change vs. Prev. Month": summary["MoM_Pct_Fmt"],
    })
    st.dataframe(pd.DataFrame(seasonal_table_dict), height="content", width="stretch", hide_index=True)

    st.caption(
        f"Sources: [U.S. EIA Monthly Retail Gasoline Prices (EMM_EPMR_PTE_NUS_DPG, XLS)]({EIA_MONTHLY_URL}) & "
        f"[BLS CPI-U via FRED (CPIAUCSL, CSV)]({FRED_CPI_URL}), "
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

    ann_prices_arr = ann["Price"].to_numpy()
    central_ann = float(np.median(ann_prices_arr)) if is_median else float(np.mean(ann_prices_arr))
    ci_low_ann, ci_high_ann = ci_func(ann_prices_arr)
    ci_ann_fmt = f"${ci_low_ann:.2f}–${ci_high_ann:.2f}"

    central_word_title = "Median" if is_median else "Mean"
    central_word_lower = "median" if is_median else "mean"

    peak_row = ann.loc[ann["Price"].idxmax()]
    low_row = ann.loc[ann["Price"].idxmin()]
    cagr = ((end_price / base_price) ** (1.0 / (end_year - start_year)) - 1.0) * 100

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

    # Pass only non-NaN columns so strict browser JSON.parse succeeds
    ann_records = ann[
        ["Year_Str", "Price", "Price_Fmt", "YoY_Diff_Fmt", "YoY_Pct_Fmt", "Pct_Start_Fmt"]
    ].to_dict(orient="records")

    # Lock y-axis domain and share an explicit band x-scale across both stacked charts
    y_max_a = float(np.ceil(ann["Price"].max() * 2) / 2 + 0.25)
    locked_y_scale_a = alt.Scale(domain=[0, y_max_a], nice=False, clamp=True)
    all_year_strs = ann["Year_Str"].tolist()
    tick_years = [str(y) for y in range(start_year, end_year + 1, 3)]

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
        opacity=0.16,
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

    points_layer = base_annual.mark_circle(size=70, color="#1f6f78", opacity=0.95).encode(
        y=alt.Y("Price:Q", scale=locked_y_scale_a),
        tooltip=[
            alt.Tooltip("Year_Str:N", title="Year"),
            alt.Tooltip("Price_Fmt:N", title="Annual Mean Price"),
            alt.Tooltip("YoY_Diff_Fmt:N", title="YoY Change ($)"),
            alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
            alt.Tooltip("Pct_Start_Fmt:N", title=f"% Change vs. {start_year}"),
        ],
    )

    # Always render the horizontal dashed reference line at the selected 34-year Mean or Median
    central_rule = (
        alt.Chart(alt.Data(values=[{"central_val": central_ann}]))
        .mark_rule(color="#555555", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(y=alt.Y("central_val:Q", scale=locked_y_scale_a))
    )

    annual_layers = []

    # Optional 95% confidence interval band around the selected 34-year Mean or Median
    if show_ci:
        ci_band_df = [{"ci_low": ci_low_ann, "ci_high": ci_high_ann}]
        ci_band_rect = (
            alt.Chart(alt.Data(values=ci_band_df))
            .mark_rect(color="#555555", opacity=0.11)
            .encode(
                y=alt.Y("ci_low:Q", scale=locked_y_scale_a),
                y2=alt.Y2("ci_high:Q"),
            )
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

    basis_tag = f"{latest_label} Dollars" if is_real else "Nominal"
    if show_ci:
        subtitle_a = (
            f"Dashed line = {len(ann)}-year {central_word_lower} (${central_ann:.2f}/gal; "
            f"shaded 95% CI: {ci_ann_fmt}, {basis_tag}) · Hover any year for details"
        )
    else:
        subtitle_a = (
            f"Dashed line = {len(ann)}-year {central_word_lower} (${central_ann:.2f}/gal, {basis_tag}) · Hover any year for details"
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
                "Event": "Emergency Strategic Petroleum Reserve (SPR) drawdown & ethanol blending mandates",
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
        ]

        events_df = pd.DataFrame(annual_events_meta)
        events_merged = pd.merge(
            events_df,
            ann[["Year", "Year_Str", "Price", "Price_Fmt", "YoY_Diff_Fmt", "YoY_Pct_Fmt"]],
            on="Year",
            how="inner",
        )
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
                    alt.Tooltip("Price_Fmt:N", title="Annual Mean Price"),
                    alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
                ],
            )
        )

        above_texts = (
            alt.Chart(alt.Data(values=[r for r in event_records if r["Above_Label"]]))
            .mark_text(dy=-11, fontSize=9.8, fontWeight="bold", color="#111111")
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

        annual_layers.extend([event_rules, event_diamonds, above_texts, below_texts])

    annual_line_chart = (
        alt.layer(*annual_layers)
        .resolve_scale(color="independent", y="shared")
        .properties(
            width=640,
            height=370,
            title=alt.TitleParams(
                text=f"Annual Mean U.S. Regular Retail Gasoline Price ({start_year}–{end_year})",
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

    # Bottom bar chart uses the exact same shared band x-scale and vconcat alignment
    yoy_all = ann.copy()
    yoy_all["YoY_Pct_Plot"] = yoy_all["YoY_Pct"].fillna(0.0)
    yoy_records = yoy_all[
        ["Year_Str", "YoY_Pct_Plot", "Direction", "Price_Fmt", "YoY_Diff_Fmt", "YoY_Pct_Fmt"]
    ].to_dict(orient="records")

    yoy_bar_chart = (
        alt.Chart(alt.Data(values=yoy_records))
        .mark_bar()
        .encode(
            x=alt.X(
                "Year_Str:O",
                title="Year",
                scale=shared_x_scale,
                axis=shared_x_axis,
            ),
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
            tooltip=[
                alt.Tooltip("Year_Str:N", title="Year"),
                alt.Tooltip("Price_Fmt:N", title="Annual Mean Price"),
                alt.Tooltip("YoY_Diff_Fmt:N", title="YoY Change ($)"),
                alt.Tooltip("YoY_Pct_Fmt:N", title="YoY Change (%)"),
            ],
        )
        .properties(
            width=640,
            height=250,
            title=alt.TitleParams(
                text=f"Year-over-Year Percentage Price Change ({start_year}–{end_year})",
                subtitle="Annual percentage swing compared to the prior calendar year (1992 baseline has no prior year)",
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

    display_annual = pd.DataFrame({
        "Year": ann["Year_Str"],
        "Mean Price": ann["Price_Fmt"],
        "YoY Change ($)": ann["YoY_Diff_Fmt"],
        "YoY Change (%)": ann["YoY_Pct_Fmt"],
        f"% Change vs. {start_year}": ann["Pct_Start_Fmt"],
    }).iloc[::-1]  # Most recent year first

    st.dataframe(display_annual, height="content", width="stretch", hide_index=True)

    st.caption(
        f"Sources: [U.S. EIA Annual Retail Gasoline Prices (EMM_EPMR_PTE_NUS_DPG, {start_year}–{end_year}, XLS)]({EIA_ANNUAL_URL}) & "
        f"[BLS CPI-U via FRED (CPIAUCSL, CSV)]({FRED_CPI_URL})."
    )

with st.expander("View Python Source Code"):
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")
