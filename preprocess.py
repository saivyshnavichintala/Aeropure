import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler
)

from Load_Data import load_data


# ============================================================
# AEROPURE
# AIR QUALITY PREPROCESSING SYSTEM
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Merged_PRSA_Data.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed_data"
)

CHART_DIR = os.path.join(
    PROCESSED_DIR,
    "preprocessing_charts"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# ============================================================
# DATASET FEATURES
# ============================================================

NUMERICAL_FEATURES = [
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

TARGET = "PM2.5"

CATEGORICAL_FEATURES = [
    "station",
    "wd"
]

TIME_FEATURES = [
    "year",
    "month",
    "day",
    "hour"
]


# ============================================================
# SAVE GRAPH FUNCTION
# ============================================================

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

    return path


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_preprocessing():

    print("\n")
    print("=" * 75)
    print("AEROPURE - COMPLETE DATA PREPROCESSING")
    print("=" * 75)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_data()

    original_rows = len(df)

    original_columns = len(df.columns)

    original_shape = df.shape

    print("\n1. DATASET OVERVIEW")
    print("-" * 60)

    print(
        "Original dataset shape:",
        original_shape
    )

    print(
        "Original rows:",
        original_rows
    )

    print(
        "Original columns:",
        original_columns
    )

    # ========================================================
    # 2. REMOVE DUPLICATE COLUMN
    # ========================================================

    print("\n2. DATA CLEANING")
    print("-" * 60)

    duplicate_column_removed = False

    if "Station" in df.columns:

        df = df.drop(
            columns=["Station"]
        )

        duplicate_column_removed = True

        print(
            "Removed duplicate column: Station"
        )

    else:

        print(
            "No duplicate Station column found."
        )

    # ========================================================
    # 3. REQUIRED COLUMN CHECK
    # ========================================================

    required_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + TIME_FEATURES
        + [TARGET]
    )

    missing_required = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_required:

        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing_required)
        )

    # ========================================================
    # 4. MISSING VALUES BEFORE
    # ========================================================

    print("\n3. MISSING VALUE ANALYSIS")
    print("-" * 60)

    missing_before = (
        df.isna()
        .sum()
    )

    missing_before_dict = {
        str(col): int(value)
        for col, value
        in missing_before.items()
        if value > 0
    }

    print(
        missing_before_dict
    )

    # ========================================================
    # 5. DUPLICATE RECORD DETECTION
    # ========================================================

    print("\n4. DUPLICATE RECORD REMOVAL")
    print("-" * 60)

    duplicate_keys = [
        "station",
        "year",
        "month",
        "day",
        "hour"
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_keys,
        keep="first"
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    df = df[
        ~duplicate_mask
    ].copy()

    print(
        "Duplicate records removed:",
        duplicate_count
    )

    # ========================================================
    # 6. MISSING VALUE HANDLING
    # ========================================================

    print("\n5. MISSING VALUE HANDLING")
    print("-" * 60)

    imputation_details = {}

    # Numerical columns
    for col in NUMERICAL_FEATURES:

        missing_count = int(
            df[col].isna().sum()
        )

        median_value = float(
            df[col].median()
        )

        df[col] = (
            df[col]
            .fillna(median_value)
        )

        imputation_details[col] = {
            "method": "Median",
            "missing": missing_count,
            "replacement": median_value
        }

    # Categorical columns
    for col in CATEGORICAL_FEATURES:

        missing_count = int(
            df[col].isna().sum()
        )

        mode_values = (
            df[col]
            .mode()
        )

        if len(mode_values) > 0:

            mode_value = str(
                mode_values.iloc[0]
            )

            df[col] = (
                df[col]
                .fillna(mode_value)
            )

            imputation_details[col] = {
                "method": "Mode",
                "missing": missing_count,
                "replacement": mode_value
            }

    # Target
    target_missing = int(
        df[TARGET].isna().sum()
    )

    df = df.dropna(
        subset=[TARGET]
    )

    print(
        "Missing PM2.5 rows removed:",
        target_missing
    )

    # ========================================================
    # 7. OUTLIER DETECTION
    # ========================================================

    print("\n6. OUTLIER DETECTION AND TREATMENT")
    print("-" * 60)

    outlier_details = {}

    all_numeric = (
        NUMERICAL_FEATURES
        + [TARGET]
    )

    total_outliers = 0

    for col in all_numeric:

        Q1 = float(
            df[col].quantile(0.25)
        )

        Q3 = float(
            df[col].quantile(0.75)
        )

        IQR = Q3 - Q1

        lower_bound = Q1 - (
            1.5 * IQR
        )

        upper_bound = Q3 + (
            1.5 * IQR
        )

        mask = (
            (df[col] < lower_bound)
            |
            (df[col] > upper_bound)
        )

        count = int(
            mask.sum()
        )

        total_outliers += count

        # Clip outliers
        df[col] = (
            df[col]
            .clip(
                lower=lower_bound,
                upper=upper_bound
            )
        )

        outlier_details[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower": lower_bound,
            "upper": upper_bound,
            "count": count
        }

        print(
            f"{col}: {count} outliers"
        )

    # ========================================================
    # 8. FEATURE ENGINEERING
    # ========================================================

    print("\n7. FEATURE ENGINEERING")
    print("-" * 60)

    df["datetime"] = pd.to_datetime(
        df[
            [
                "year",
                "month",
                "day",
                "hour"
            ]
        ]
    )

    df["day_of_week"] = (
        df["datetime"]
        .dt.dayofweek
    )

    df["day_of_year"] = (
        df["datetime"]
        .dt.dayofyear
    )

    df["is_weekend"] = (
        df["day_of_week"]
        >= 5
    ).astype(int)

    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        elif month in [3, 4, 5]:
            return "Spring"

        elif month in [6, 7, 8]:
            return "Summer"

        return "Autumn"

    df["season"] = (
        df["month"]
        .apply(get_season)
    )

    feature_engineering = [
        "datetime",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "season"
    ]

    # ========================================================
    # 9. AIR QUALITY CATEGORY
    # ========================================================

    print("\n8. AIR QUALITY CATEGORY")
    print("-" * 60)

    def classify_air_quality(value):

        if value <= 35:
            return "Good"

        elif value <= 75:
            return "Moderate"

        elif value <= 150:
            return "Unhealthy"

        else:
            return "Very Unhealthy"

    df[
        "air_quality_category"
    ] = (
        df[TARGET]
        .apply(
            classify_air_quality
        )
    )

    air_quality_counts = (
        df[
            "air_quality_category"
        ]
        .value_counts()
    )

    print(
        air_quality_counts
    )

    # ========================================================
    # 10. CORRELATION ANALYSIS
    # ========================================================

    print("\n9. CORRELATION ANALYSIS")
    print("-" * 60)

    correlation_columns = (
        NUMERICAL_FEATURES
        + [TARGET]
    )

    correlation_matrix = (
        df[
            correlation_columns
        ]
        .corr()
    )

    correlation_file = os.path.join(
        PROCESSED_DIR,
        "correlation_matrix.csv"
    )

    correlation_matrix.to_csv(
        correlation_file
    )

    target_correlation = (
        correlation_matrix[TARGET]
        .drop(TARGET)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(
        "\nCorrelation with PM2.5:"
    )

    print(
        target_correlation
    )

    # ========================================================
    # 11. CORRELATION GRAPH
    # ========================================================

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        correlation_matrix,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(
            len(
                correlation_matrix.columns
            )
        ),
        correlation_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(
            len(
                correlation_matrix.columns
            )
        ),
        correlation_matrix.columns
    )

    plt.title(
        "AeroPure Correlation Heatmap"
    )

    correlation_chart = save_chart(
        "correlation_heatmap.png"
    )

    # ========================================================
    # 12. TARGET CORRELATION GRAPH
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    target_correlation.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Feature Correlation with PM2.5"
    )

    plt.xlabel(
        "Correlation Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    target_correlation_chart = save_chart(
        "target_correlation.png"
    )

    # ========================================================
    # 13. TRAIN TEST SPLIT
    # ========================================================

    print("\n10. TRAIN-TEST SPLIT")
    print("-" * 60)

    # 2013-2016 -> Training
    # 2017       -> Testing

    train_df = df[
        df["year"] < 2017
    ].copy()

    test_df = df[
        df["year"] == 2017
    ].copy()

    train_rows_before = len(
        train_df
    )

    test_rows_before = len(
        test_df
    )

    print(
        "Training rows:",
        train_rows_before
    )

    print(
        "Testing rows:",
        test_rows_before
    )

    # ========================================================
    # 14. ONE-HOT ENCODING
    # ========================================================

    print("\n11. ONE-HOT ENCODING")
    print("-" * 60)

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_categories = (
        train_df[
            CATEGORICAL_FEATURES
        ]
        .astype(str)
    )

    test_categories = (
        test_df[
            CATEGORICAL_FEATURES
        ]
        .astype(str)
    )

    train_encoded = (
        encoder.fit_transform(
            train_categories
        )
    )

    test_encoded = (
        encoder.transform(
            test_categories
        )
    )

    onehot_names = (
        encoder
        .get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )

    train_onehot = pd.DataFrame(
        train_encoded,
        columns=onehot_names,
        index=train_df.index
    )

    test_onehot = pd.DataFrame(
        test_encoded,
        columns=onehot_names,
        index=test_df.index
    )

    train_df = pd.concat(
        [
            train_df,
            train_onehot
        ],
        axis=1
    )

    test_df = pd.concat(
        [
            test_df,
            test_onehot
        ],
        axis=1
    )

    onehot_count = len(
        onehot_names
    )

    print(
        "One-Hot Features:",
        onehot_count
    )

    # ========================================================
    # ONE-HOT GRAPH
    # ========================================================

    onehot_counts = (
        train_onehot
        .sum()
        .sort_values(
            ascending=False
        )
    )

    # Display only top 20 for readability
    top_onehot = (
        onehot_counts
        .head(20)
    )

    plt.figure(
        figsize=(12, 7)
    )

    top_onehot.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "One-Hot Encoding - Top Categories"
    )

    plt.xlabel(
        "Number of Records"
    )

    plt.ylabel(
        "Encoded Feature"
    )

    onehot_chart = save_chart(
        "one_hot_encoding.png"
    )

    # ========================================================
    # 15. ORDINAL ENCODING
    # ========================================================

    print("\n12. ORDINAL ENCODING")
    print("-" * 60)

    ordinal_categories = [
        [
            "Good",
            "Moderate",
            "Unhealthy",
            "Very Unhealthy"
        ]
    ]

    ordinal_encoder = OrdinalEncoder(
        categories=ordinal_categories
    )

    train_ordinal = (
        ordinal_encoder.fit_transform(
            train_df[
                ["air_quality_category"]
            ]
        )
    )

    test_ordinal = (
        ordinal_encoder.transform(
            test_df[
                ["air_quality_category"]
            ]
        )
    )

    train_df[
        "air_quality_ordinal"
    ] = train_ordinal.astype(int)

    test_df[
        "air_quality_ordinal"
    ] = test_ordinal.astype(int)

    ordinal_mapping = {
        "Good": 0,
        "Moderate": 1,
        "Unhealthy": 2,
        "Very Unhealthy": 3
    }

    # ========================================================
    # ORDINAL GRAPH
    # ========================================================

    ordinal_counts = (
        train_df[
            "air_quality_ordinal"
        ]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(8, 5)
    )

    ordinal_counts.plot(
        kind="bar"
    )

    plt.title(
        "Ordinal Encoding of Air Quality"
    )

    plt.xlabel(
        "Ordinal Value"
    )

    plt.ylabel(
        "Number of Records"
    )

    ordinal_chart = save_chart(
        "ordinal_encoding.png"
    )

    # ========================================================
    # 16. STANDARDIZATION
    # ========================================================

    print("\n13. STANDARDIZATION")
    print("-" * 60)

    scaler_standard = StandardScaler()

    train_standard = (
        scaler_standard.fit_transform(
            train_df[
                NUMERICAL_FEATURES
            ]
        )
    )

    test_standard = (
        scaler_standard.transform(
            test_df[
                NUMERICAL_FEATURES
            ]
        )
    )

    standard_stats = {}

    for i, feature in enumerate(
        NUMERICAL_FEATURES
    ):

        train_df[
            feature
            + "_standardized"
        ] = train_standard[:, i]

        test_df[
            feature
            + "_standardized"
        ] = test_standard[:, i]

        standard_stats[feature] = {
            "mean": float(
                train_standard[:, i].mean()
            ),
            "std": float(
                train_standard[:, i].std()
            )
        }

    # ========================================================
    # STANDARDIZATION GRAPH
    # ========================================================

    sample_size = min(
        10000,
        len(train_df)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        train_df[
            "PM10"
        ].iloc[:sample_size],
        bins=50,
        alpha=0.6,
        label="Original PM10"
    )

    plt.hist(
        train_df[
            "PM10_standardized"
        ].iloc[:sample_size],
        bins=50,
        alpha=0.6,
        label="Standardized PM10"
    )

    plt.title(
        "Original vs Standardized PM10"
    )

    plt.xlabel(
        "Value"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()

    standard_chart = save_chart(
        "standardization_comparison.png"
    )

    # ========================================================
    # 17. MIN-MAX NORMALIZATION
    # ========================================================

    print("\n14. MIN-MAX NORMALIZATION")
    print("-" * 60)

    scaler_minmax = MinMaxScaler()

    train_minmax = (
        scaler_minmax.fit_transform(
            train_df[
                NUMERICAL_FEATURES
            ]
        )
    )

    test_minmax = (
        scaler_minmax.transform(
            test_df[
                NUMERICAL_FEATURES
            ]
        )
    )

    minmax_stats = {}

    for i, feature in enumerate(
        NUMERICAL_FEATURES
    ):

        train_df[
            feature
            + "_minmax"
        ] = train_minmax[:, i]

        test_df[
            feature
            + "_minmax"
        ] = test_minmax[:, i]

        minmax_stats[feature] = {
            "min": float(
                train_minmax[:, i].min()
            ),
            "max": float(
                train_minmax[:, i].max()
            )
        }

    # ========================================================
    # MINMAX GRAPH
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        train_df[
            "PM10_minmax"
        ].iloc[:sample_size],
        bins=50
    )

    plt.title(
        "PM10 after Min-Max Normalization"
    )

    plt.xlabel(
        "Normalized Value (0-1)"
    )

    plt.ylabel(
        "Frequency"
    )

    minmax_chart = save_chart(
        "minmax_normalization.png"
    )

    # ========================================================
    # 18. STANDARDIZATION VS MINMAX
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        train_df[
            "PM10_standardized"
        ].iloc[:sample_size],
        bins=50,
        alpha=0.6,
        label="Standardization"
    )

    plt.hist(
        train_df[
            "PM10_minmax"
        ].iloc[:sample_size],
        bins=50,
        alpha=0.6,
        label="Min-Max"
    )

    plt.title(
        "Standardization vs Min-Max Normalization"
    )

    plt.xlabel(
        "Transformed Value"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()

    scaling_comparison_chart = save_chart(
        "scaling_comparison.png"
    )

    # ========================================================
    # 19. FEATURE SELECTION
    # ========================================================

    print("\n15. FEATURE SELECTION")
    print("-" * 60)

    # Select features according to their correlation
    # with PM2.5.

    selected_features = (
        target_correlation
        .abs()
        .sort_values(
            ascending=False
        )
        .head(8)
        .index
        .tolist()
    )

    print(
        "Selected features:"
    )

    for feature in selected_features:
        print(
            " -",
            feature
        )

    # Feature selection graph

    selected_values = (
        target_correlation[
            selected_features
        ]
        .sort_values()
    )

    plt.figure(
        figsize=(10, 6)
    )

    selected_values.plot(
        kind="barh"
    )

    plt.title(
        "Selected Features Based on PM2.5 Correlation"
    )

    plt.xlabel(
        "Correlation with PM2.5"
    )

    plt.ylabel(
        "Feature"
    )

    feature_selection_chart = save_chart(
        "feature_selection.png"
    )

    # ========================================================
    # 20. MISSING VALUES AFTER
    # ========================================================

    print("\n16. FINAL MISSING VALUE CHECK")
    print("-" * 60)

    missing_after = (
        df.isna()
        .sum()
    )

    missing_after_dict = {
        str(col): int(value)
        for col, value
        in missing_after.items()
        if value > 0
    }

    print(
        "Remaining missing values:"
    )

    print(
        missing_after_dict
    )

    # ========================================================
    # 21. DATA LEAKAGE PREVENTION
    # ========================================================

    leakage_message = (
        "The dataset was split chronologically before "
        "fitting the encoders and scalers. StandardScaler, "
        "MinMaxScaler and OneHotEncoder were fitted only "
        "using training data. Test data was transformed "
        "using those already-fitted objects."
    )

    print("\n17. DATA LEAKAGE PREVENTION")
    print("-" * 60)

    print(
        leakage_message
    )

    # ========================================================
    # 22. EXPORT
    # ========================================================

    print("\n18. PROCESSED DATA EXPORT")
    print("-" * 60)

    train_file = os.path.join(
        PROCESSED_DIR,
        "AeroPure_Train.csv"
    )

    test_file = os.path.join(
        PROCESSED_DIR,
        "AeroPure_Test.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    print(
        "Training file:"
    )

    print(
        train_file
    )

    print(
        "Testing file:"
    )

    print(
        test_file
    )

    # ========================================================
    # 23. FINAL RESULT
    # ========================================================

    result = {

        # Overview
        "original_rows":
            int(original_rows),

        "original_columns":
            int(original_columns),

        "remaining_rows":
            int(len(df)),

        "remaining_columns":
            int(len(df.columns)),

        # Cleaning
        "duplicate_column_removed":
            duplicate_column_removed,

        "duplicate_count":
            duplicate_count,

        # Missing
        "missing_before":
            missing_before_dict,

        "missing_after":
            missing_after_dict,

        "imputation_details":
            imputation_details,

        "target_missing_removed":
            target_missing,

        # Outliers
        "outliers":
            outlier_details,

        "total_outliers":
            total_outliers,

        # Feature engineering
        "feature_engineering":
            feature_engineering,

        # Air quality
        "air_quality_counts": {
            str(k): int(v)
            for k, v
            in air_quality_counts.items()
        },

        # Correlation
        "target_correlation": {
            str(k): float(v)
            for k, v
            in target_correlation.items()
        },

        # Features
        "numerical_features":
            NUMERICAL_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "target":
            TARGET,

        # One-hot
        "onehot_count":
            onehot_count,

        "onehot_features": [
            str(x)
            for x in onehot_names
        ],

        # Ordinal
        "ordinal_mapping":
            ordinal_mapping,

        # Scaling
        "standard_stats":
            standard_stats,

        "minmax_stats":
            minmax_stats,

        # Feature selection
        "selected_features":
            selected_features,

        # Train/test
        "train_rows":
            int(train_rows_before),

        "test_rows":
            int(test_rows_before),

        "train_columns":
            int(len(train_df.columns)),

        "test_columns":
            int(len(test_df.columns)),

        # Leakage
        "leakage_message":
            leakage_message,

        # Files
        "train_file":
            train_file,

        "test_file":
            test_file,

        "correlation_file":
            correlation_file,

        # Charts
        "charts": {
            "correlation":
                correlation_chart,

            "target_correlation":
                target_correlation_chart,

            "onehot":
                onehot_chart,

            "ordinal":
                ordinal_chart,

            "standardization":
                standard_chart,

            "minmax":
                minmax_chart,

            "scaling_comparison":
                scaling_comparison_chart,

            "feature_selection":
                feature_selection_chart
        }
    }

    print("\n")
    print("=" * 75)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 75)

    return result


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    try:

        result = run_preprocessing()

        print("\nFINAL SUMMARY")
        print("=" * 75)

        print(
            "Original rows:",
            result["original_rows"]
        )

        print(
            "Remaining rows:",
            result["remaining_rows"]
        )

        print(
            "Duplicate records removed:",
            result["duplicate_count"]
        )

        print(
            "Total outliers:",
            result["total_outliers"]
        )

        print(
            "Training rows:",
            result["train_rows"]
        )

        print(
            "Testing rows:",
            result["test_rows"]
        )

        print(
            "One-Hot Features:",
            result["onehot_count"]
        )

        print(
            "\nSelected Features:"
        )

        for feature in result[
            "selected_features"
        ]:

            print(
                " -",
                feature
            )

        print(
            "\nTarget Correlation:"
        )

        for feature, value in (
            result[
                "target_correlation"
            ].items()
        ):

            print(
                f"{feature}: {value:.4f}"
            )

        print(
            "\nAll preprocessing completed."
        )

    except Exception as e:

        print("\nPREPROCESSING ERROR")
        print("=" * 75)
        print(e)
        raise