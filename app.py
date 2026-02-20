"""
NYC Taxi Trip Dashboard - Main Application
==========================================
Run with: streamlit run app.py

"""

import streamlit as st
import polars as pl
from datetime import datetime, date
import requests
from pathlib import Path
import plotly.express as px


st.set_page_config(
    page_title="NYC Taxi Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A5F; margin-bottom: 0; }
    .sub-header  { font-size: 1.1rem; color: #666; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# Constants

TRIP_URL  = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL  = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
DATA_DIR  = Path("data/raw")
TRIP_FILE = DATA_DIR / "yellow_tripdata_2024-01.parquet"
ZONE_FILE = DATA_DIR / "taxi_zone_lookup.csv"

TRIP_COLS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount",
    "payment_type", "total_amount",
]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

BOROUGH_COLORS = {
    "Manhattan":    "#E74C3C",
    "Queens":       "#3498DB",
    "Brooklyn":     "#2ECC71",
    "Bronx":        "#F39C12",
    "EWR":          "#9B59B6",
    "Staten Island":"#1ABC9C",
}


# Helpers

def _to_date(val) -> date:
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    return date.fromisoformat(str(val)[:10])


def download_file(url: str, destination: Path, chunk_size: int = 65_536) -> bool:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total    = int(response.headers.get("content-length", 0))
        received = 0
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r  {received/total*100:.1f}%  ({received/1e6:.1f}/{total/1e6:.1f} MB)", end="")
        print(f"\n  Saved -> {destination.name}  ({received/1e6:.2f} MB)")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


# Data loading

@st.cache_resource(show_spinner="Loading trip data…")
def _load_lazy() -> pl.LazyFrame:
    if not TRIP_FILE.exists():
        download_file(TRIP_URL, TRIP_FILE)

    lf = (
        pl.scan_parquet(TRIP_FILE)
        .select(TRIP_COLS)

        #  cleaning

        .drop_nulls(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                             "PULocationID", "DOLocationID", "fare_amount"])
        .filter(pl.col("trip_distance") > 0)
        .filter((pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 500))
        .filter(pl.col("tpep_dropoff_datetime") > pl.col("tpep_pickup_datetime"))
        .filter(
            (pl.col("tpep_pickup_datetime") >= datetime(2024, 1, 1)) &
            (pl.col("tpep_pickup_datetime") <  datetime(2024, 2, 1))
        )

        #  feature engineering

        .with_columns([
            ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
             .dt.total_seconds() / 60).alias("trip_duration_min"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.strftime("%A").alias("pickup_day_of_week"),
            (
                pl.when(pl.col("payment_type") == 1).then(pl.lit("Credit Card"))
                  .when(pl.col("payment_type") == 2).then(pl.lit("Cash"))
                  .when(pl.col("payment_type") == 3).then(pl.lit("No Charge"))
                  .when(pl.col("payment_type") == 4).then(pl.lit("Dispute"))
                  .otherwise(pl.lit("Unknown"))
            ).alias("payment_method"),
        ])
        .with_columns([
            (pl.col("trip_distance") / (pl.col("trip_duration_min") / 60))
            .alias("trip_speed_mph")
        ])
    )
    return lf


@st.cache_resource(show_spinner="Loading zone data…")
def _load_zones() -> pl.DataFrame:
    if not ZONE_FILE.exists():
        download_file(ZONE_URL, ZONE_FILE)
    return pl.read_csv(ZONE_FILE)


# Core filtered LazyFrame builder

def _filtered_lazy(
    start_date,
    end_date,
    selected_days: tuple,
    hour_range: tuple,
    selected_payments: tuple,
) -> pl.LazyFrame:

    start_dt = datetime.combine(_to_date(start_date), datetime.min.time())
    end_dt   = datetime.combine(_to_date(end_date),   datetime.max.time())

    return (
        _load_lazy()
        .filter(
            (pl.col("tpep_pickup_datetime") >= start_dt) &
            (pl.col("tpep_pickup_datetime") <= end_dt) &
            (pl.col("pickup_day_of_week").is_in(list(selected_days))) &
            (pl.col("pickup_hour").is_between(hour_range[0], hour_range[1])) &
            (pl.col("payment_method").is_in(list(selected_payments)))
        )
    )


# Aggregation functions

@st.cache_data(show_spinner=False)
def get_key_metrics(start_date, end_date, days, hours, payments):
    return (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .select([
            pl.len().alias("total_trips"),
            pl.col("fare_amount").mean().alias("avg_fare"),
            pl.col("trip_distance").mean().alias("avg_distance"),
            pl.col("trip_duration_min").mean().alias("avg_duration"),
            pl.col("total_amount").sum().alias("total_revenue"),
        ])
        .collect()
        .row(0, named=True)
    )


@st.cache_data(show_spinner=False)
def get_date_coverage(start_date, end_date, days, hours, payments):
    return (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .select([
            pl.col("tpep_pickup_datetime").min().alias("mn"),
            pl.col("tpep_pickup_datetime").max().alias("mx"),
        ])
        .collect()
        .row(0, named=True)
    )


@st.cache_data(show_spinner=False)
def get_top_payment(start_date, end_date, days, hours, payments):
    return (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .group_by("payment_method")
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
        .limit(1)
        .collect()
        .row(0, named=True)["payment_method"]
    )


@st.cache_data(show_spinner=False)
def get_top_pickup_zones(start_date, end_date, days, hours, payments) -> pl.DataFrame:
    zones = _load_zones().select(["LocationID", "Zone", "Borough"])
    result = (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .group_by("PULocationID")
        .agg(pl.len().alias("total_trips"))
        .collect()
        .join(zones, left_on="PULocationID", right_on="LocationID", how="inner")
        .sort("total_trips", descending=True)
        .head(10)
        .select([pl.col("Zone").alias("pickup_zone"), "Borough", "total_trips"])
    )
    return result


@st.cache_data(show_spinner=False)
def get_hourly_fare(start_date, end_date, days, hours, payments) -> pl.DataFrame:
    return (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .group_by("pickup_hour")
        .agg([
            pl.col("fare_amount").mean().round(2).alias("avg_fare"),
            pl.len().alias("total_trips"),
        ])
        .sort("pickup_hour")
        .collect()
    )


@st.cache_data(show_spinner=False)
def get_payment_distribution(start_date, end_date, days, hours, payments) -> pl.DataFrame:
    return (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .group_by("payment_method")
        .agg(pl.len().alias("total_trips"))
        .collect()
        .with_columns(
            (pl.col("total_trips") * 100 / pl.col("total_trips").sum())
            .round(2).alias("percentage")
        )
        .sort("total_trips", descending=True)
    )


@st.cache_data(show_spinner=False)
def get_heatmap_data(start_date, end_date, days, hours, payments) -> pl.DataFrame:
    return (
        _filtered_lazy(start_date, end_date, days, hours, payments)
        .group_by(["pickup_day_of_week", "pickup_hour"])
        .agg(pl.len().alias("trip_count"))
        .collect()
    )


@st.cache_data(show_spinner=False)
def get_distance_data(start_date, end_date, days, hours, payments):

    base = _filtered_lazy(start_date, end_date, days, hours, payments).select("trip_distance")
    histogram_df = base.filter(pl.col("trip_distance") <= 15).collect()
    median_val   = base.select(pl.col("trip_distance").median()).collect().item()
    return histogram_df, median_val


# Load base data

try:
    _load_lazy()
    zone_df = _load_zones()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Collect min/max dates for sidebar widgets

@st.cache_data
def _date_bounds():
    return (
        _load_lazy()
        .select([
            pl.col("tpep_pickup_datetime").min().alias("mn"),
            pl.col("tpep_pickup_datetime").max().alias("mx"),
        ])
        .collect()
        .row(0, named=True)
    )


@st.cache_data
def _payment_options():
    return sorted(
        _load_lazy()
        .select("payment_method")
        .unique()
        .collect()["payment_method"]
        .to_list()
    )


bounds          = _date_bounds()
min_date        = _to_date(bounds["mn"])
max_date        = _to_date(bounds["mx"])
payment_options = _payment_options()
ALL_DAYS        = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# Header

st.markdown('<p class="main-header">NYC Taxi Trip Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Exploring Yellow Taxi Data from January 2024</p>', unsafe_allow_html=True)
st.divider()

st.subheader("About This Dashboard")
st.markdown("""
This dashboard lets you explore NYC Yellow Taxi trip data. Use the **sidebar** to apply
filters and dive deeper into the data. Highlights:

- **Top Pickup Zones** - which areas had the most taxi pickups?
- **Average Fare by Hour** - how do fares change throughout the day?
- **Trip Distance Distribution** - are most trips short or long?
- **Payment Type Breakdown** - what payment methods do riders prefer?
- **Trips Heatmap** - when are taxis busiest during the week?

Built with Streamlit & Plotly for COMP 3610 A1.
""")
st.divider()


# Sidebar Filters

st.sidebar.header("Filters")

#  Date range
st.sidebar.subheader("Date Range")

if "start_date" not in st.session_state:
    st.session_state.start_date = min_date
if "end_date" not in st.session_state:
    st.session_state.end_date = max_date

st.session_state.start_date = _to_date(st.session_state.start_date)
st.session_state.end_date   = _to_date(st.session_state.end_date)

date_range = st.sidebar.date_input(
    "Select Date Range (January 2024)",
    value=(st.session_state.start_date, st.session_state.end_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range_input",
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = _to_date(date_range[0]), _to_date(date_range[1])
else:
    start_date = end_date = _to_date(date_range)

st.session_state.start_date = start_date
st.session_state.end_date   = end_date

st.sidebar.markdown("---")

#  Day of week
st.sidebar.subheader("Day of Week")

if "selected_days" not in st.session_state:
    st.session_state.selected_days = ALL_DAYS

selected_days = st.sidebar.multiselect(
    "Select Day(s) of Week", ALL_DAYS,
    default=st.session_state.selected_days,
    key="day_filter",
)
st.session_state.selected_days = selected_days

st.sidebar.markdown("---")

#  Hour range
st.sidebar.subheader("Time of Day")

if "hour_range" not in st.session_state:
    st.session_state.hour_range = (0, 23)

hour_range = st.sidebar.slider(
    "Select Hour Range", 0, 23,
    value=st.session_state.hour_range,
    key="hour_slider",
)
st.session_state.hour_range = hour_range

st.sidebar.markdown("---")

#  Payment type
st.sidebar.subheader("Payment Type")

if "selected_payments" not in st.session_state:
    st.session_state.selected_payments = payment_options

selected_payments = st.sidebar.multiselect(
    "Select Payment Type(s)", payment_options,
    default=st.session_state.selected_payments,
    key="payment_filter",
)
st.session_state.selected_payments = selected_payments

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** NYC Yellow Taxi (Jan 2024)")
st.sidebar.markdown("---")


# Guard - at least one day and payment selected

if not selected_days or not selected_payments:
    st.warning("Please select at least one day and payment type.")
    st.stop()

# Build filter key (used by every cached aggregation)
fkey = dict(
    start_date = start_date,
    end_date   = end_date,
    days       = tuple(selected_days),
    hours      = tuple(hour_range),
    payments   = tuple(selected_payments),
)


# Key Metrics

st.subheader("Key Metrics at a Glance")

with st.spinner("Computing metrics…"):
    m = get_key_metrics(**fkey)

if m["total_trips"] == 0:
    st.warning("No trips match the selected filters. Try widening your criteria.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
with col1: st.metric("Total Trips",   f"{m['total_trips']:,}",        help="Trips matching your filters")
with col2: st.metric("Average Fare",  f"${m['avg_fare']:.2f}",        help="Mean fare amount")
with col3: st.metric("Avg Distance",  f"{m['avg_distance']:.2f} mi",  help="Average trip distance")
with col4: st.metric("Avg Duration",  f"{m['avg_duration']:.1f} min", help="Average trip duration")
with col5: st.metric("Total Revenue", f"${m['total_revenue']:,.2f}",  help="Total amount collected")

st.divider()


# Data Coverage

st.subheader("Data Coverage")
col1, col2 = st.columns(2)

with col1:
    cov = get_date_coverage(**fkey)
    mn  = _to_date(cov["mn"])
    mx  = _to_date(cov["mx"])
    st.info(f"**Filtered Date Range:** {mn} to {mx}")

with col2:
    top_pay = get_top_payment(**fkey)
    st.info(f"**Most Common Payment:** {top_pay}")

st.divider()


# Visualizations

st.subheader("Visualizations (Plotly Charts)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top Pickup Zones",
    "Average Fare by Hour",
    "Trip Distance Distribution",
    "Payment Type Breakdown",
    "Trips Heatmap",
])


#  Tab 1 - Top Pickup Zones
with tab1:
    st.subheader("Top 10 Pickup Zones by Trip Count")
    try:
        with st.spinner("Loading pickup zones…"):
            r1 = get_top_pickup_zones(**fkey)

        fig = px.bar(
            r1, x="total_trips", y="pickup_zone",
            color="Borough", orientation="h",
            color_discrete_map=BOROUGH_COLORS,
            text="total_trips",
            labels={"total_trips": "Number of Trips", "pickup_zone": "Pickup Zone"},
        )

        fig.update_layout(
            height=600,
            yaxis={"categoryorder": "total ascending"},
            legend=dict(title="Borough", x=0.85, y=0.15),
        )

        fig.update_traces(texttemplate="%{text:,}", textposition="outside", textfont_size=9)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
**Key Insights:**
- Pickup activity is heavily concentrated in Manhattan, accounting for eight of the top ten zones.
- Midtown Center and Upper East Side South lead in trip volume with nearly identical counts.
- Airports in Queens (JFK and LaGuardia) remain major pickup hubs, highlighting the strong influence of air travel.
""")
    except Exception as e:
        st.error(f"Error loading pickup zones: {e}")


#  Tab 2 - Average Fare by Hour
with tab2:
    with st.spinner("Computing hourly fares…"):
        r2 = get_hourly_fare(**fkey)

    fig = px.line(
        r2, x="pickup_hour", y="avg_fare",
        title="Average Taxi Fare by Hour of Day",
        labels={"pickup_hour": "Hour of Day", "avg_fare": "Average Fare ($)"},
        markers=True,
    )

    fig.update_traces(
        hovertemplate="<b>Hour:</b> %{x}:00<br><b>Avg Fare:</b> $%{y:.2f}<br><extra></extra>",
        line=dict(color="#E74C3C", width=3),
        marker=dict(size=8),
    )

    fig.update_layout(
        height=500, hovermode="x unified",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(tickprefix="$"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key Insights:**
- Average fares peak at $27.50 during 5 AM, likely reflecting early morning surcharges and airport runs.
- Fares stabilise through late morning and afternoon, hovering around $18.
- A gradual increase reappears late at night, indicating elevated demand during evening travel hours.
""")


#  Tab 3 - Trip Distance Distribution
with tab3:
    st.subheader("Distribution of Trip Distances (0-15 miles)")

    with st.spinner("Building histogram…"):
        dist_df, median_dist = get_distance_data(**fkey)

    fig = px.histogram(
        dist_df, x="trip_distance", nbins=60,
        title="Histogram: Distribution of Trip Distances (0-15 miles)",
        labels={"trip_distance": "Trip Distance (miles)"},
        opacity=0.75,
    )

    fig.add_vline(
        x=median_dist, line_dash="dash", line_color="red",
        annotation_text=f"Median: {median_dist:.2f} mi",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Trip Distance (miles)", yaxis_title="Frequency",
        template="plotly_white", height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key Insights:**
- Most trips are short - the majority are under 3 miles with a median of 1.7 miles.
- The right-skewed distribution confirms that long-distance trips are the exception, not the rule.
- This reflects NYC taxi usage as primarily intra-borough rather than cross-city travel.
""")


#  Tab 4 - Payment Type Breakdown
with tab4:
    st.subheader("Payment Type Distribution")

    with st.spinner("Computing payment breakdown…"):
        pay = get_payment_distribution(**fkey)

    fig = px.bar(
        pay, x="payment_method", y="total_trips",
        title="Payment Type Distribution",
        color="payment_method",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="percentage",
    )

    fig.update_traces(texttemplate="%{text}%", textposition="outside")

    fig.update_layout(
        xaxis_title="Payment Method", yaxis_title="Total Trips",
        showlegend=False, height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key Insights:**
- Credit card dominates at 80.09% of trips, suggesting most riders prefer cashless transactions.
- Cash is a distant second; No Charge, Dispute, and Unknown combined account for only 5.17%.
- Disputes and No Charge are rare but worth monitoring for fare errors or data quality issues.
""")


#  Tab 5 - Trips Heatmap
with tab5:
    st.subheader("Taxi Trip Volume: Hour of Day vs Day of Week")

    with st.spinner("Building heatmap…"):
        hmap_pd = get_heatmap_data(**fkey).to_pandas()

    pivot = (
        hmap_pd
        .pivot(index="pickup_day_of_week", columns="pickup_hour", values="trip_count")
        .fillna(0)
        .reindex(index=WEEKDAY_ORDER, columns=list(range(24)), fill_value=0)
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Trip Count"),
        x=list(range(24)), y=WEEKDAY_ORDER,
        color_continuous_scale="YlOrRd",
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key Patterns:**
- **Weekday evenings (17-18 h) are peak demand periods**, with Tuesday-Thursday highest, likely driven by evening commutes.
- **Early morning (0-6 h) is consistently the quietest**, except Saturday-Sunday nights which reflect weekend nightlife activity.
""")