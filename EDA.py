import os
import warnings

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from Load_Data import load_data


warnings.filterwarnings("ignore")


# ============================================================
# AEROPURE - EXPLORATORY DATA ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(CHART_DIR, exist_ok=True)


def save_chart(filename):

    path = os.path.join(
        CHART_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()

    return f"/static/charts/{filename}"


def run_eda():

    df = load_data()

    rows = len(df)
    columns = len(df.columns)

    duplicates = int(
        df.duplicated().sum()
    )

    missing = df.isna().sum()

    charts = []

    # ========================================================
    # REMOVE DUPLICATE COLUMN IF PRESENT
    # ========================================================

    if "Station" in df.columns:

        df = df.drop(
            columns=["Station"]
        )

    # ========================================================
    # NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM"
    ]

    available_numeric = [
        col for col in numeric_columns
        if col in df.columns
    ]

    # ========================================================
    # SAMPLE DATA
    # ========================================================

    if len(df) > 50000:

        sample_df = df.sample(
            50000,
            random_state=42
        )

    else:

        sample_df = df.copy()

    # ========================================================
    # 1. MISSING VALUES
    # ========================================================

    plt.figure(figsize=(12, 6))

    missing_plot = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing_plot) > 0:

        missing_plot.plot(
            kind="bar"
        )

        plt.title(
            "Missing Values by Column"
        )

        plt.xlabel(
            "Columns"
        )

        plt.ylabel(
            "Number of Missing Values"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

    else:

        plt.text(
            0.5,
            0.5,
            "No Missing Values",
            ha="center",
            va="center"
        )

    charts.append(
        save_chart(
            "missing_values.png"
        )
    )

    # ========================================================
    # 2. MISSING VALUE HEATMAP
    # ========================================================

    plt.figure(figsize=(14, 6))

    heatmap_sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    sns.heatmap(
        heatmap_sample.isnull(),
        cbar=False,
        yticklabels=False
    )

    plt.title(
        "Missing Value Heatmap"
    )

    charts.append(
        save_chart(
            "missing_heatmap.png"
        )
    )

    # ========================================================
    # 3. NUMERIC DISTRIBUTIONS
    # ========================================================

    if len(available_numeric) > 0:

        sample_df[
            available_numeric
        ].hist(
            figsize=(16, 12),
            bins=30
        )

        plt.suptitle(
            "Numeric Feature Distributions"
        )

        charts.append(
            save_chart(
                "numeric_distribution.png"
            )
        )

    # ========================================================
    # 4. PM2.5 DISTRIBUTION
    # ========================================================

    if "PM2.5" in df.columns:

        plt.figure(figsize=(10, 6))

        sns.histplot(
            sample_df["PM2.5"].dropna(),
            bins=50,
            kde=True
        )

        plt.title(
            "PM2.5 Distribution"
        )

        plt.xlabel(
            "PM2.5"
        )

        plt.ylabel(
            "Frequency"
        )

        charts.append(
            save_chart(
                "pm25_distribution.png"
            )
        )

    # ========================================================
    # 5. BOXPLOTS
    # ========================================================

    for column in available_numeric:

        plt.figure(figsize=(10, 5))

        sns.boxplot(
            x=sample_df[column]
        )

        plt.title(
            f"{column} Boxplot"
        )

        plt.xlabel(
            column
        )

        filename = (
            column
            .replace(".", "")
            .replace(" ", "_")
            .lower()
            + "_boxplot.png"
        )

        charts.append(
            save_chart(filename)
        )

    # ========================================================
    # 6. PM2.5 vs TEMPERATURE
    # ========================================================

    if (
        "PM2.5" in df.columns
        and "TEMP" in df.columns
    ):

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=sample_df,
            x="TEMP",
            y="PM2.5",
            alpha=0.3
        )

        plt.title(
            "PM2.5 vs Temperature"
        )

        plt.xlabel(
            "Temperature"
        )

        plt.ylabel(
            "PM2.5"
        )

        charts.append(
            save_chart(
                "pm25_temperature.png"
            )
        )

    # ========================================================
    # 7. PM2.5 vs PRESSURE
    # ========================================================

    if (
        "PM2.5" in df.columns
        and "PRES" in df.columns
    ):

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=sample_df,
            x="PRES",
            y="PM2.5",
            alpha=0.3
        )

        plt.title(
            "PM2.5 vs Pressure"
        )

        plt.xlabel(
            "Pressure"
        )

        plt.ylabel(
            "PM2.5"
        )

        charts.append(
            save_chart(
                "pm25_pressure.png"
            )
        )

    # ========================================================
    # 8. PM2.5 vs WIND SPEED
    # ========================================================

    if (
        "PM2.5" in df.columns
        and "WSPM" in df.columns
    ):

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=sample_df,
            x="WSPM",
            y="PM2.5",
            alpha=0.3
        )

        plt.title(
            "PM2.5 vs Wind Speed"
        )

        plt.xlabel(
            "Wind Speed"
        )

        plt.ylabel(
            "PM2.5"
        )

        charts.append(
            save_chart(
                "pm25_windspeed.png"
            )
        )

    # ========================================================
    # 9. STATION DISTRIBUTION
    # ========================================================

    if "station" in df.columns:

        plt.figure(figsize=(14, 6))

        df["station"].value_counts().plot(
            kind="bar"
        )

        plt.title(
            "Records by Monitoring Station"
        )

        plt.xlabel(
            "Station"
        )

        plt.ylabel(
            "Number of Records"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        charts.append(
            save_chart(
                "station_distribution.png"
            )
        )

    # ========================================================
    # 10. STATION-WISE PM2.5
    # ========================================================

    if (
        "station" in df.columns
        and "PM2.5" in df.columns
    ):

        station_pm = (
            df.groupby("station")["PM2.5"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        plt.figure(figsize=(14, 6))

        station_pm.plot(
            kind="bar"
        )

        plt.title(
            "Average PM2.5 by Station"
        )

        plt.xlabel(
            "Station"
        )

        plt.ylabel(
            "Average PM2.5"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        charts.append(
            save_chart(
                "station_pm25.png"
            )
        )

    # ========================================================
    # 11. MONTHLY PM2.5
    # ========================================================

    if (
        "month" in df.columns
        and "PM2.5" in df.columns
    ):

        monthly_pm = (
            df.groupby("month")["PM2.5"]
            .mean()
        )

        plt.figure(figsize=(10, 6))

        monthly_pm.plot(
            marker="o"
        )

        plt.title(
            "Average PM2.5 by Month"
        )

        plt.xlabel(
            "Month"
        )

        plt.ylabel(
            "Average PM2.5"
        )

        charts.append(
            save_chart(
                "monthly_pm25.png"
            )
        )

    # ========================================================
    # 12. HOURLY PM2.5
    # ========================================================

    if (
        "hour" in df.columns
        and "PM2.5" in df.columns
    ):

        hourly_pm = (
            df.groupby("hour")["PM2.5"]
            .mean()
        )

        plt.figure(figsize=(10, 6))

        hourly_pm.plot(
            marker="o"
        )

        plt.title(
            "Average PM2.5 by Hour"
        )

        plt.xlabel(
            "Hour"
        )

        plt.ylabel(
            "Average PM2.5"
        )

        charts.append(
            save_chart(
                "hourly_pm25.png"
            )
        )

    # ========================================================
    # 13. WIND DIRECTION
    # ========================================================

    if "wd" in df.columns:

        plt.figure(figsize=(12, 6))

        df["wd"].value_counts().plot(
            kind="bar"
        )

        plt.title(
            "Wind Direction Distribution"
        )

        plt.xlabel(
            "Wind Direction"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        charts.append(
            save_chart(
                "wind_direction.png"
            )
        )

    # ========================================================
    # 14. CORRELATION HEATMAP
    # ========================================================

    if len(available_numeric) > 1:

        correlation = df[
            available_numeric
        ].corr()

        plt.figure(
            figsize=(12, 9)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0
        )

        plt.title(
            "Correlation Heatmap"
        )

        charts.append(
            save_chart(
                "correlation_heatmap.png"
            )
        )

    # ========================================================
    # 15. PAIRPLOT
    # ========================================================

    pair_columns = [
        col
        for col in [
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "TEMP"
        ]
        if col in df.columns
    ]

    if len(pair_columns) >= 2:

        pair_df = sample_df[
            pair_columns
        ].dropna()

        pair_df = pair_df.sample(
            min(2000, len(pair_df)),
            random_state=42
        )

        sns.pairplot(
            pair_df
        )

        pairplot_path = os.path.join(
            CHART_DIR,
            "pairplot.png"
        )

        plt.savefig(
            pairplot_path,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close("all")

        charts.append(
            "/static/charts/pairplot.png"
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "rows": rows,

        "columns": columns,

        "duplicates": duplicates,

        "missing": missing.to_dict(),

        "charts": charts
    }


if __name__ == "__main__":

    result = run_eda()

    print("=" * 70)
    print("AEROPURE - EDA")
    print("=" * 70)

    print(
        "Rows:",
        result["rows"]
    )

    print(
        "Columns:",
        result["columns"]
    )

    print(
        "Duplicates:",
        result["duplicates"]
    )

    print(
        "\nCharts generated:",
        len(result["charts"])
    )

    for chart in result["charts"]:
        print(chart)

    print("\nEDA COMPLETED SUCCESSFULLY")