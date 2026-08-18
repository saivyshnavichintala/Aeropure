import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from Load_Data import load_data


CHARTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static",
    "charts"
)


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_eda():

    data = load_data()

    charts = []

    sns.set_style("whitegrid")

    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    rows = len(data)
    columns = len(data.columns)

    duplicates = int(data.duplicated().sum())

    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x=missing.index,
            y=missing.values
        )

        plt.xticks(rotation=45, ha="right")

        plt.title("Missing Values by Column")
        plt.ylabel("Number of Missing Values")

        plt.tight_layout()

        plt.savefig(
            _chart_path("missing_values.png")
        )

        plt.close()

        charts.append("missing_values.png")

        # Missing heatmap

        plt.figure(figsize=(14, 6))

        sns.heatmap(
            data.isnull(),
            cbar=False,
            yticklabels=False
        )

        plt.title("Missing Value Heatmap")

        plt.tight_layout()

        plt.savefig(
            _chart_path("missing_heatmap.png")
        )

        plt.close()

        charts.append("missing_heatmap.png")

    # =====================================================
    # NUMERIC DISTRIBUTIONS
    # =====================================================

    numeric_columns = [
        "PM2.5",
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir"
    ]

    numeric_columns = [
        col for col in numeric_columns
        if col in data.columns
    ]

    if numeric_columns:

        data[numeric_columns].hist(
            figsize=(14, 10),
            bins=25
        )

        plt.suptitle(
            "Air Quality Numeric Distributions"
        )

        plt.tight_layout()

        plt.savefig(
            _chart_path("numeric_distribution.png")
        )

        plt.close()

        charts.append("numeric_distribution.png")

    # =====================================================
    # PM2.5 DISTRIBUTION
    # =====================================================

    if "PM2.5" in data.columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data["PM2.5"].dropna(),
            kde=True
        )

        mean_value = data["PM2.5"].mean()

        plt.axvline(
            mean_value,
            linestyle="--",
            label=f"Mean = {mean_value:.2f}"
        )

        plt.title("PM2.5 Distribution")

        plt.xlabel("PM2.5")
        plt.ylabel("Frequency")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            _chart_path("pm25_distribution.png")
        )

        plt.close()

        charts.append("pm25_distribution.png")

    # =====================================================
    # BOXPLOTS
    # =====================================================

    box_columns = [
        "PM2.5",
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir"
    ]

    box_columns = [
        col for col in box_columns
        if col in data.columns
    ]

    for col in box_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            x=data[col].dropna()
        )

        plt.title(f"{col} Boxplot")
        plt.xlabel(col)

        plt.tight_layout()

        filename = (
            f"boxplot_{col.replace('.', '_')}.png"
        )

        plt.savefig(
            _chart_path(filename)
        )

        plt.close()

        charts.append(filename)

    # =====================================================
    # PM2.5 VS TEMPERATURE
    # =====================================================

    if "PM2.5" in data.columns and "TEMP" in data.columns:

        plt.figure(figsize=(8, 6))

        sns.regplot(
            data=data,
            x="TEMP",
            y="PM2.5",
            scatter_kws={"alpha": 0.4}
        )

        plt.title("PM2.5 vs Temperature")

        plt.tight_layout()

        plt.savefig(
            _chart_path("pm25_temperature.png")
        )

        plt.close()

        charts.append("pm25_temperature.png")

    # =====================================================
    # PM2.5 VS PRESSURE
    # =====================================================

    if "PM2.5" in data.columns and "PRES" in data.columns:

        plt.figure(figsize=(8, 6))

        sns.regplot(
            data=data,
            x="PRES",
            y="PM2.5",
            scatter_kws={"alpha": 0.4}
        )

        plt.title("PM2.5 vs Pressure")

        plt.tight_layout()

        plt.savefig(
            _chart_path("pm25_pressure.png")
        )

        plt.close()

        charts.append("pm25_pressure.png")

    # =====================================================
    # PM2.5 VS WIND SPEED
    # =====================================================

    if "PM2.5" in data.columns and "Iws" in data.columns:

        plt.figure(figsize=(8, 6))

        sns.regplot(
            data=data,
            x="Iws",
            y="PM2.5",
            scatter_kws={"alpha": 0.4}
        )

        plt.title("PM2.5 vs Wind Speed")

        plt.tight_layout()

        plt.savefig(
            _chart_path("pm25_windspeed.png")
        )

        plt.close()

        charts.append("pm25_windspeed.png")

    # =====================================================
    # STATION DISTRIBUTION
    # =====================================================

    if "station" in data.columns:

        plt.figure(figsize=(10, 5))

        order = data["station"].value_counts().index

        sns.countplot(
            data=data,
            x="station",
            order=order
        )

        plt.title("Records by Station")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            _chart_path("station_distribution.png")
        )

        plt.close()

        charts.append("station_distribution.png")

    # =====================================================
    # STATION VS PM2.5
    # =====================================================

    if "station" in data.columns and "PM2.5" in data.columns:

        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=data,
            x="station",
            y="PM2.5"
        )

        plt.title("PM2.5 Levels by Station")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            _chart_path("station_pm25.png")
        )

        plt.close()

        charts.append("station_pm25.png")

    # =====================================================
    # MONTHLY PM2.5
    # =====================================================

    if "month" in data.columns and "PM2.5" in data.columns:

        monthly = data.groupby(
            "month"
        )["PM2.5"].mean()

        plt.figure(figsize=(9, 5))

        sns.lineplot(
            x=monthly.index,
            y=monthly.values,
            marker="o"
        )

        plt.title("Average PM2.5 by Month")

        plt.xlabel("Month")
        plt.ylabel("Average PM2.5")

        plt.tight_layout()

        plt.savefig(
            _chart_path("monthly_pm25.png")
        )

        plt.close()

        charts.append("monthly_pm25.png")

    # =====================================================
    # HOURLY PM2.5
    # =====================================================

    if "hour" in data.columns and "PM2.5" in data.columns:

        hourly = data.groupby(
            "hour"
        )["PM2.5"].mean()

        plt.figure(figsize=(10, 5))

        sns.lineplot(
            x=hourly.index,
            y=hourly.values,
            marker="o"
        )

        plt.title("Average PM2.5 by Hour")

        plt.xlabel("Hour")
        plt.ylabel("Average PM2.5")

        plt.xticks(range(24))

        plt.tight_layout()

        plt.savefig(
            _chart_path("hourly_pm25.png")
        )

        plt.close()

        charts.append("hourly_pm25.png")

    # =====================================================
    # WIND DIRECTION
    # =====================================================

    if "cbwd" in data.columns:

        plt.figure(figsize=(8, 5))

        order = data["cbwd"].value_counts().index

        sns.countplot(
            data=data,
            x="cbwd",
            order=order
        )

        plt.title("Wind Direction Distribution")

        plt.tight_layout()

        plt.savefig(
            _chart_path("wind_direction.png")
        )

        plt.close()

        charts.append("wind_direction.png")

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    numeric_data = data.select_dtypes(
        include="number"
    )

    if numeric_data.shape[1] >= 2:

        plt.figure(figsize=(12, 9))

        correlation = numeric_data.corr()

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(
            _chart_path("correlation_heatmap.png")
        )

        plt.close()

        charts.append("correlation_heatmap.png")

    # =====================================================
    # PAIR PLOT
    # =====================================================

    pair_columns = [
        "PM2.5",
        "DEWP",
        "TEMP",
        "PRES",
        "Iws"
    ]

    pair_columns = [
        col for col in pair_columns
        if col in data.columns
    ]

    if len(pair_columns) >= 2:

        pair_data = data[
            pair_columns
        ].dropna()

        if len(pair_data) > 3000:

            pair_data = pair_data.sample(
                3000,
                random_state=42
            )

        pair = sns.pairplot(pair_data)

        pair.savefig(
            _chart_path("pairplot.png")
        )

        plt.close("all")

        charts.append("pairplot.png")

    # =====================================================
    # RETURN
    # =====================================================

    return {
        "rows": rows,
        "columns": columns,
        "duplicates": duplicates,
        "missing": missing.to_dict(),
        "charts": charts
    }