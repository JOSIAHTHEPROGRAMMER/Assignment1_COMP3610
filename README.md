# COMP 3610 Assignment 1 - NYC Taxi Trip Dashboard

An interactive Streamlit dashboard analyzing NYC Yellow Taxi trip data for January 2024. Built with Polars for data processing and Plotly for visualizations.

---

## Live Dashboard

[Deployed Link](https://a1comp3610-8jb5kpxuh9pjkzshmqzko2.streamlit.app/)

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher (I used Python 3.11.9)
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/JOSIAHTHEPROGRAMMER/Assignment1_COMP3610
cd Assignment1_COMP3610
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
polars==1.31.0
duckdb==1.3.2
plotly==5.24.1
streamlit==1.53.1
requests==2.32.4
```

3. Run the dashboard:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Data

NYC Taxi and Limousine Commission (TLC) Yellow Taxi Trip Records for January 2024.

- Yellow Taxi Trip Data: Parquet file with approximately 3 million trip records
- Taxi Zone Lookup Table: CSV file mapping location IDs to borough and zone names

The app downloads both files automatically on first run.

---

## Features

- Top 10 pickup zones by trip volume
- Average fare by hour of day
- Trip distance distribution
- Payment type breakdown
- Trip volume heatmap by hour and day of week
- Sidebar filters for day, hour range, and payment type
