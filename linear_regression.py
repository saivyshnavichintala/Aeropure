import os
import pickle
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Merged_PRSA_Data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model_results"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "linear_regression_charts"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# ============================================================
# GLOBAL CACHE
# ============================================================

_ANALYSIS_CACHE = None


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            "Dataset file not found:\n"
            + DATA_PATH
        )

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "Station": str
        },
        low_memory=False
    )

    print("Dataset loaded successfully.")

    return df


# ============================================================
# CREATE DATETIME FEATURES
# ============================================================

def create_time_features(df):

    df = df.copy()

    date_columns = [
        "year",
        "month",
        "day",
        "hour"
    ]

    for column in date_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["datetime"] = pd.to_datetime(
        df[
            [
                "year",
                "month",
                "day",
                "hour"
            ]
        ],
        errors="coerce"
    )

    invalid_datetime_rows = int(
        df["datetime"].isna().sum()
    )

    if invalid_datetime_rows > 0:

        df = df.dropna(
            subset=["datetime"]
        ).copy()

    # Day of week
    df["day_of_week"] = (
        df["datetime"].dt.dayofweek
    )

    # Day of year
    df["day_of_year"] = (
        df["datetime"].dt.dayofyear
    )

    # Week number
    df["week_of_year"] = (
        df["datetime"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Weekend
    df["is_weekend"] = (
        df["datetime"]
        .dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    # Season
    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        elif month in [3, 4, 5]:
            return "Spring"

        elif month in [6, 7, 8]:
            return "Summer"

        else:
            return "Autumn"

    df["season"] = (
        df["month"]
        .apply(get_season)
    )

    return df, invalid_datetime_rows


# ============================================================
# CREATE ONE HOT ENCODER
# ============================================================

def create_encoder():

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

    return encoder


# ============================================================
# TRAIN ALL THREE MODELS
# ============================================================

def train_all_models():

    global _ANALYSIS_CACHE

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if _ANALYSIS_CACHE is not None:

        print(
            "\nUsing cached regression results..."
        )

        return _ANALYSIS_CACHE

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_dataset()

    raw_rows = int(df.shape[0])
    raw_columns = int(df.shape[1])

    raw_column_names = list(
        df.columns
    )

    print("\nRaw dataset shape:")
    print(
        raw_rows,
        "rows x",
        raw_columns,
        "columns"
    )

    # --------------------------------------------------------
    # EXACT DUPLICATE REMOVAL
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Remove exact duplicates BEFORE dropping
    # uppercase Station column.
    #
    # This prevents accidental removal of valid
    # station-hour records.
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # DROP DUPLICATE STATION COLUMN
    # --------------------------------------------------------

    if "Station" in df.columns:

        df = df.drop(
            columns=["Station"]
        )

    # --------------------------------------------------------
    # CREATE TIME FEATURES
    # --------------------------------------------------------

    df, invalid_datetime_rows = (
        create_time_features(df)
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target = "PM2.5"

    if target not in df.columns:

        raise ValueError(
            "Target column PM2.5 was not found."
        )

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    numeric_features = [

        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",

        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM",

        "year",
        "month",
        "day",
        "hour",

        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend"
    ]

    categorical_features = [

        "wd",
        "season",
        "station"
    ]

    all_features = (
        numeric_features
        + categorical_features
    )

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in all_features
        if column not in df.columns
    ]

    if len(missing_columns) > 0:

        raise ValueError(
            "Required feature columns are missing:\n"
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # CONVERT NUMERIC FEATURES
    # --------------------------------------------------------

    for column in numeric_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # MISSING TARGET
    # --------------------------------------------------------

    missing_target_rows = int(
        df[target].isna().sum()
    )

    df = df.dropna(
        subset=[target]
    ).copy()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    station_count = int(
        df["station"].nunique(
            dropna=True
        )
    )

    date_min = df["datetime"].min()
    date_max = df["datetime"].max()

    # --------------------------------------------------------
    # FEATURE DATA
    # --------------------------------------------------------

    X = df[
        all_features
    ].copy()

    y = df[
        target
    ].copy()

    # --------------------------------------------------------
    # CHRONOLOGICAL TRAIN TEST SPLIT
    # --------------------------------------------------------
    #
    # Training:
    # 2013 - 2016
    #
    # Testing:
    # 2017
    #
    # No random splitting is used because this is
    # time-series air-quality data.
    # --------------------------------------------------------

    train_mask = (
        df["datetime"]
        < pd.Timestamp("2017-01-01")
    )

    test_mask = (
        df["datetime"]
        >= pd.Timestamp("2017-01-01")
    )

    X_train = X.loc[
        train_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()

    test_metadata = df.loc[
        test_mask,
        [
            "datetime",
            "station"
        ]
    ].copy()

    # --------------------------------------------------------
    # TRAIN / TEST CHECK
    # --------------------------------------------------------

    if len(X_train) == 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if len(X_test) == 0:

        raise ValueError(
            "Testing dataset is empty."
        )

    # --------------------------------------------------------
    # MISSING VALUES BEFORE IMPUTATION
    # --------------------------------------------------------

    numeric_missing_train = int(
        X_train[
            numeric_features
        ].isna().sum().sum()
    )

    categorical_missing_train = int(
        X_train[
            categorical_features
        ].isna().sum().sum()
    )

    # --------------------------------------------------------
    # NUMERIC PIPELINE
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                create_encoder()
            )
        ]
    )

    # --------------------------------------------------------
    # COLUMN TRANSFORMER
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],

        sparse_threshold=1.0
    )

    # --------------------------------------------------------
    # FIT PREPROCESSOR
    # --------------------------------------------------------

    print(
        "\nApplying preprocessing..."
    )

    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_encoded = (
        preprocessor.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # STANDARD SCALING
    # --------------------------------------------------------
    #
    # with_mean=False is required for sparse matrices.
    # --------------------------------------------------------

    scaler = StandardScaler(
        with_mean=False
    )

    X_train_processed = (
        scaler.fit_transform(
            X_train_encoded
        )
    )

    X_test_processed = (
        scaler.transform(
            X_test_encoded
        )
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    feature_names = [
        name.replace(
            "numeric__",
            ""
        ).replace(
            "categorical__",
            ""
        )
        for name in feature_names
    ]

    encoded_feature_count = len(
        feature_names
    )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    models = {

        "linear": LinearRegression(),

        "ridge": Ridge(
            alpha=1.0,
            solver="lsqr"
        ),

        "lasso": Lasso(
            alpha=0.01,
            max_iter=20000,
            tol=0.0001,
            selection="cyclic"
        )
    }

    model_names = {

        "linear":
            "Linear Regression",

        "ridge":
            "Ridge Regression",

        "lasso":
            "Lasso Regression"
    }

    # --------------------------------------------------------
    # MODEL CONFIG DETAILS
    # --------------------------------------------------------

    model_configurations = {

        "linear": {

            "algorithm":
                "Ordinary Least Squares Linear Regression",

            "regularization":
                "None",

            "penalty":
                "No penalty",

            "alpha":
                "Not applicable",

            "max_iterations":
                "Default",

            "description":
                "Baseline linear regression without regularization."
        },

        "ridge": {

            "algorithm":
                "Ridge Regression",

            "regularization":
                "L2 regularization",

            "penalty":
                "L2",

            "alpha":
                "1.0",

            "max_iterations":
                "Solver: LSQR",

            "description":
                "Reduces large coefficients using L2 regularization."
        },

        "lasso": {

            "algorithm":
                "Lasso Regression",

            "regularization":
                "L1 regularization",

            "penalty":
                "L1",

            "alpha":
                "0.01",

            "max_iterations":
                "20000",

            "description":
                "Can shrink some coefficients exactly to zero."
        }
    }

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    model_results = {}

    fitted_models = {}

    print("\n")
    print("=" * 70)
    print("TRAINING THREE REGRESSION MODELS")
    print("=" * 70)

    for model_key, model in models.items():

        print(
            "\nTraining:",
            model_names[model_key]
        )

        # Fit
        model.fit(
            X_train_processed,
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test_processed
        )

        # Metrics
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # Coefficients
        coefficients = (
            model.coef_
        )

        coefficients = np.asarray(
            coefficients
        ).ravel()

        coefficient_table = pd.DataFrame({

            "feature":
                feature_names,

            "coefficient":
                coefficients

        })

        coefficient_table[
            "absolute_coefficient"
        ] = (
            coefficient_table[
                "coefficient"
            ].abs()
        )

        coefficient_table[
            "direction"
        ] = np.where(
            coefficient_table[
                "coefficient"
            ] >= 0,
            "Positive",
            "Negative"
        )

        coefficient_table = (
            coefficient_table
            .sort_values(
                "absolute_coefficient",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        non_zero_coefficients = int(
            np.sum(
                np.abs(coefficients)
                > 1e-8
            )
        )

        # Store
        model_results[
            model_key
        ] = {

            "name":
                model_names[model_key],

            "mae":
                float(mae),

            "mse":
                float(mse),

            "rmse":
                float(rmse),

            "r2":
                float(r2),

            "predictions":
                predictions,

            "coefficients":
                coefficient_table,

            "intercept":
                float(
                    model.intercept_
                ),

            "non_zero_coefficients":
                non_zero_coefficients
        }

        fitted_models[
            model_key
        ] = model

        print(
            "MAE :",
            round(mae, 4)
        )

        print(
            "MSE :",
            round(mse, 4)
        )

        print(
            "RMSE:",
            round(rmse, 4)
        )

        print(
            "R2  :",
            round(r2, 4)
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison_rows = []

    for model_key in [
        "linear",
        "ridge",
        "lasso"
    ]:

        result = model_results[
            model_key
        ]

        comparison_rows.append({

            "key":
                model_key,

            "model":
                result["name"],

            "MAE":
                result["mae"],

            "MSE":
                result["mse"],

            "RMSE":
                result["rmse"],

            "R2":
                result["r2"],

            "non_zero_coefficients":
                result[
                    "non_zero_coefficients"
                ]
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------
    #
    # Lowest RMSE is selected as best.
    # --------------------------------------------------------

    best_model_key = (
        comparison_df
        .sort_values(
            "RMSE",
            ascending=True
        )
        .iloc[0]["key"]
    )

    best_model_key = str(
        best_model_key
    )

    best_model_name = (
        model_names[
            best_model_key
        ]
    )

    # --------------------------------------------------------
    # CORRELATION TABLE
    # --------------------------------------------------------
    #
    # Pearson correlation between original numeric
    # predictors and PM2.5.
    #
    # Categorical variables are not included.
    # --------------------------------------------------------

    correlation_columns = (
        numeric_features
        + [target]
    )

    correlation_df = (
        df.loc[
            train_mask,
            correlation_columns
        ]
        .corr()
        [target]
        .drop(target)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
        .reset_index()
    )

    correlation_df.columns = [
        "feature",
        "correlation"
    ]

    def correlation_strength(value):

        absolute_value = abs(
            value
        )

        if absolute_value >= 0.70:

            return "Strong"

        elif absolute_value >= 0.40:

            return "Moderate"

        elif absolute_value >= 0.20:

            return "Weak"

        else:

            return "Very Weak"

    correlation_df[
        "direction"
    ] = np.where(
        correlation_df[
            "correlation"
        ] >= 0,
        "Positive",
        "Negative"
    )

    correlation_df[
        "strength"
    ] = correlation_df[
        "correlation"
    ].apply(
        correlation_strength
    )

    # --------------------------------------------------------
    # SAVE CORRELATION
    # --------------------------------------------------------

    correlation_path = os.path.join(
        MODEL_DIR,
        "correlation_table.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE MODEL COMPARISON
    # --------------------------------------------------------

    comparison_path = os.path.join(
        MODEL_DIR,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    for model_key in fitted_models:

        model_path = os.path.join(
            MODEL_DIR,
            model_key
            + "_regression_model.pkl"
        )

        model_package = {

            "model":
                fitted_models[
                    model_key
                ],

            "preprocessor":
                preprocessor,

            "scaler":
                scaler,

            "feature_names":
                feature_names,

            "target":
                target
        }

        with open(
            model_path,
            "wb"
        ) as file:

            pickle.dump(
                model_package,
                file
            )

    # --------------------------------------------------------
    # PREPROCESSING SUMMARY
    # --------------------------------------------------------

    preprocessing_summary = {

        "Raw rows":
            raw_rows,

        "Raw columns":
            raw_columns,

        "Exact duplicate rows removed":
            duplicate_rows,

        "Invalid datetime rows removed":
            invalid_datetime_rows,

        "Missing PM2.5 rows removed":
            missing_target_rows,

        "Training rows":
            int(len(X_train)),

        "Testing rows":
            int(len(X_test)),

        "Numeric features":
            len(numeric_features),

        "Categorical features":
            len(categorical_features),

        "Original features":
            len(all_features),

        "Encoded features":
            encoded_feature_count,

        "Numeric missing values in training":
            numeric_missing_train,

        "Categorical missing values in training":
            categorical_missing_train,

        "Numeric imputation":
            "Median",

        "Categorical imputation":
            "Most Frequent",

        "Categorical encoding":
            "One-Hot Encoding",

        "Feature scaling":
            "StandardScaler",

        "Data split":
            "Chronological: before 2017 = training, 2017 = testing"
    }

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    dataset_information = {

        "Dataset":
            "Beijing Multi-Site Air Quality Dataset",

        "File":
            "Merged_PRSA_Data.csv",

        "Raw rows":
            raw_rows,

        "Raw columns":
            raw_columns,

        "Rows after exact duplicate removal":
            int(len(df)),

        "Number of stations":
            station_count,

        "Date range":
            str(
                date_min
            )
            + " to "
            + str(
                date_max
            ),

        "Training period":
            "2013-2016",

        "Testing period":
            "2017",

        "Target":
            "PM2.5",

        "Problem type":
            "Regression"
    }

    # --------------------------------------------------------
    # FEATURE TARGET INFORMATION
    # --------------------------------------------------------

    feature_target_information = {

        "Target":
            target,

        "Target description":
            "Fine particulate matter concentration (PM2.5)",

        "Total original features":
            len(all_features),

        "Numeric feature count":
            len(numeric_features),

        "Categorical feature count":
            len(categorical_features),

        "Feature list":
            all_features
    }

    # --------------------------------------------------------
    # CACHE EVERYTHING
    # --------------------------------------------------------

    _ANALYSIS_CACHE = {

        "models":
            fitted_models,

        "model_results":
            model_results,

        "model_names":
            model_names,

        "model_configurations":
            model_configurations,

        "comparison_df":
            comparison_df,

        "best_model_key":
            best_model_key,

        "best_model_name":
            best_model_name,

        "correlation_df":
            correlation_df,

        "test_actual":
            np.asarray(
                y_test
            ),

        "test_metadata":
            test_metadata,

        "dataset_information":
            dataset_information,

        "feature_target_information":
            feature_target_information,

        "preprocessing_summary":
            preprocessing_summary,

        "feature_names":
            feature_names
    }

    print("\n")
    print("=" * 70)
    print("ALL THREE MODELS TRAINED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nBest Model:",
        best_model_name
    )

    return _ANALYSIS_CACHE


# ============================================================
# CHART: ACTUAL VS PREDICTED
# ============================================================

def create_actual_vs_predicted_chart(
    actual,
    predicted,
    model_key
):

    file_name = (
        "actual_vs_predicted_"
        + model_key
        + ".png"
    )

    file_path = os.path.join(
        CHART_DIR,
        file_name
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        actual,
        predicted,
        alpha=0.35,
        s=12
    )

    minimum = min(
        np.min(actual),
        np.min(predicted)
    )

    maximum = max(
        np.max(actual),
        np.max(predicted)
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linewidth=2
    )

    plt.xlabel(
        "Actual PM2.5"
    )

    plt.ylabel(
        "Predicted PM2.5"
    )

    plt.title(
        "Actual vs Predicted PM2.5 - "
        + model_key.upper()
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        file_path,
        dpi=150
    )

    plt.close()

    return (
        "linear_regression_charts/"
        + file_name
    )


# ============================================================
# CHART: RESIDUAL ANALYSIS
# ============================================================

def create_residual_chart(
    actual,
    predicted,
    model_key
):

    file_name = (
        "residual_analysis_"
        + model_key
        + ".png"
    )

    file_path = os.path.join(
        CHART_DIR,
        file_name
    )

    residuals = (
        actual
        - predicted
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        predicted,
        residuals,
        alpha=0.35,
        s=12
    )

    plt.axhline(
        0,
        linewidth=2
    )

    plt.xlabel(
        "Predicted PM2.5"
    )

    plt.ylabel(
        "Residual (Actual - Predicted)"
    )

    plt.title(
        "Residual Analysis - "
        + model_key.upper()
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        file_path,
        dpi=150
    )

    plt.close()

    return (
        "linear_regression_charts/"
        + file_name
    )


# ============================================================
# CHART: FEATURE COEFFICIENTS
# ============================================================

def create_coefficient_chart(
    coefficient_table,
    model_key
):

    file_name = (
        "feature_coefficients_"
        + model_key
        + ".png"
    )

    file_path = os.path.join(
        CHART_DIR,
        file_name
    )

    plot_df = (
        coefficient_table
        .head(15)
        .sort_values(
            "coefficient"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["coefficient"]
    )

    plt.xlabel(
        "Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Feature Coefficients - "
        + model_key.upper()
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        file_path,
        dpi=150
    )

    plt.close()

    return (
        "linear_regression_charts/"
        + file_name
    )


# ============================================================
# CHART: CORRELATION
# ============================================================

def create_correlation_chart(
    correlation_df
):

    file_name = (
        "correlation_with_pm25.png"
    )

    file_path = os.path.join(
        CHART_DIR,
        file_name
    )

    plot_df = (
        correlation_df
        .sort_values(
            "correlation"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["correlation"]
    )

    plt.xlabel(
        "Pearson Correlation"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Feature Correlation with PM2.5"
    )

    plt.axvline(
        0,
        linewidth=1.5
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        file_path,
        dpi=150
    )

    plt.close()

    return (
        "linear_regression_charts/"
        + file_name
    )


# ============================================================
# BUILD PREDICTION TABLE
# ============================================================

def build_prediction_table(
    actual,
    predicted,
    metadata
):

    table = metadata.copy()

    table["actual"] = (
        np.asarray(actual)
    )

    table["predicted"] = (
        np.asarray(predicted)
    )

    table["error"] = (
        table["actual"]
        - table["predicted"]
    )

    table["absolute_error"] = (
        table["error"].abs()
    )

    table["datetime"] = (
        pd.to_datetime(
            table["datetime"]
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    table["actual"] = (
        table["actual"]
        .round(4)
    )

    table["predicted"] = (
        table["predicted"]
        .round(4)
    )

    table["error"] = (
        table["error"]
        .round(4)
    )

    table["absolute_error"] = (
        table["absolute_error"]
        .round(4)
    )

    return table.head(50).to_dict(
        orient="records"
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def build_error_analysis(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    errors = (
        actual
        - predicted
    )

    underprediction_count = int(
        np.sum(
            errors > 0
        )
    )

    overprediction_count = int(
        np.sum(
            errors < 0
        )
    )

    total = len(
        errors
    )

    underprediction_percentage = (

        underprediction_count
        / total
        * 100
    )

    overprediction_percentage = (

        overprediction_count
        / total
        * 100
    )

    return {

        "Mean Error":
            float(
                np.mean(errors)
            ),

        "Mean Absolute Error":
            float(
                np.mean(
                    np.abs(errors)
                )
            ),

        "Maximum Underprediction":
            float(
                np.max(errors)
            ),

        "Maximum Overprediction":
            float(
                np.min(errors)
            ),

        "Standard Deviation of Error":
            float(
                np.std(errors)
            ),

        "Underprediction Count":
            underprediction_count,

        "Underprediction Percentage":
            float(
                underprediction_percentage
            ),

        "Overprediction Count":
            overprediction_count,

        "Overprediction Percentage":
            float(
                overprediction_percentage
            ),

        "Total Test Predictions":
            total
    }


# ============================================================
# SAVE SELECTED PREDICTIONS
# ============================================================

def save_selected_predictions(
    actual,
    predicted,
    metadata,
    model_key
):

    table = metadata.copy()

    table["actual"] = actual

    table["predicted"] = predicted

    table["error"] = (
        table["actual"]
        - table["predicted"]
    )

    table["absolute_error"] = (
        table["error"].abs()
    )

    file_path = os.path.join(
        MODEL_DIR,
        "predictions_"
        + model_key
        + ".csv"
    )

    table.to_csv(
        file_path,
        index=False
    )


# ============================================================
# MAIN REGRESSION FUNCTION
# ============================================================

def run_regression_analysis(
    model_type="linear"
):

    model_type = str(
        model_type
    ).lower()

    # --------------------------------------------------------
    # VALID MODEL
    # --------------------------------------------------------

    if model_type not in [
        "linear",
        "ridge",
        "lasso"
    ]:

        model_type = "linear"

    # --------------------------------------------------------
    # TRAIN / LOAD CACHE
    # --------------------------------------------------------

    analysis = (
        train_all_models()
    )

    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    selected_result = (
        analysis[
            "model_results"
        ][
            model_type
        ]
    )

    selected_model_name = (
        selected_result[
            "name"
        ]
    )

    actual = (
        analysis[
            "test_actual"
        ]
    )

    predicted = (
        selected_result[
            "predictions"
        ]
    )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    actual_predicted_chart = (
        create_actual_vs_predicted_chart(
            actual,
            predicted,
            model_type
        )
    )

    residual_chart = (
        create_residual_chart(
            actual,
            predicted,
            model_type
        )
    )

    coefficient_chart = (
        create_coefficient_chart(
            selected_result[
                "coefficients"
            ],
            model_type
        )
    )

    correlation_chart = (
        create_correlation_chart(
            analysis[
                "correlation_df"
            ]
        )
    )

    # --------------------------------------------------------
    # PREDICTION TABLE
    # --------------------------------------------------------

    prediction_table = (
        build_prediction_table(
            actual,
            predicted,
            analysis[
                "test_metadata"
            ]
        )
    )

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    error_analysis = (
        build_error_analysis(
            actual,
            predicted
        )
    )

    # --------------------------------------------------------
    # COEFFICIENT TABLE
    # --------------------------------------------------------

    coefficient_table = (
        selected_result[
            "coefficients"
        ]
        .head(30)
        .copy()
    )

    coefficient_table[
        "coefficient"
    ] = coefficient_table[
        "coefficient"
    ].round(6)

    coefficient_table[
        "absolute_coefficient"
    ] = coefficient_table[
        "absolute_coefficient"
    ].round(6)

    coefficient_table = (
        coefficient_table
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # CORRELATION TABLE
    # --------------------------------------------------------

    correlation_table = (
        analysis[
            "correlation_df"
        ]
        .copy()
    )

    correlation_table[
        "correlation"
    ] = correlation_table[
        "correlation"
    ].round(6)

    correlation_table = (
        correlation_table
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    evaluation = {

        "MAE":
            round(
                selected_result[
                    "mae"
                ],
                6
            ),

        "MSE":
            round(
                selected_result[
                    "mse"
                ],
                6
            ),

        "RMSE":
            round(
                selected_result[
                    "rmse"
                ],
                6
            ),

        "R2":
            round(
                selected_result[
                    "r2"
                ],
                6
            )
    }

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    model_comparison = []

    for row in (
        analysis[
            "comparison_df"
        ].to_dict(
            orient="records"
        )
    ):

        model_comparison.append({

            "model":
                row["model"],

            "MAE":
                round(
                    float(
                        row["MAE"]
                    ),
                    6
                ),

            "MSE":
                round(
                    float(
                        row["MSE"]
                    ),
                    6
                ),

            "RMSE":
                round(
                    float(
                        row["RMSE"]
                    ),
                    6
                ),

            "R2":
                round(
                    float(
                        row["R2"]
                    ),
                    6
                ),

            "non_zero_coefficients":
                int(
                    row[
                        "non_zero_coefficients"
                    ]
                ),

            "selected":
                row["key"]
                == model_type,

            "key":
                row["key"]
        })

    # --------------------------------------------------------
    # REGULARIZATION COMPARISON
    # --------------------------------------------------------

    regularization_comparison = [

        {

            "model":
                "Linear Regression",

            "regularization":
                "None",

            "penalty":
                "None",

            "alpha":
                "N/A",

            "RMSE":
                round(
                    analysis[
                        "model_results"
                    ]["linear"]["rmse"],
                    6
                ),

            "R2":
                round(
                    analysis[
                        "model_results"
                    ]["linear"]["r2"],
                    6
                ),

            "non_zero":
                analysis[
                    "model_results"
                ]["linear"][
                    "non_zero_coefficients"
                ]
        },

        {

            "model":
                "Ridge Regression",

            "regularization":
                "L2",

            "penalty":
                "L2 penalty",

            "alpha":
                "1.0",

            "RMSE":
                round(
                    analysis[
                        "model_results"
                    ]["ridge"]["rmse"],
                    6
                ),

            "R2":
                round(
                    analysis[
                        "model_results"
                    ]["ridge"]["r2"],
                    6
                ),

            "non_zero":
                analysis[
                    "model_results"
                ]["ridge"][
                    "non_zero_coefficients"
                ]
        },

        {

            "model":
                "Lasso Regression",

            "regularization":
                "L1",

            "penalty":
                "L1 penalty",

            "alpha":
                "0.01",

            "RMSE":
                round(
                    analysis[
                        "model_results"
                    ]["lasso"]["rmse"],
                    6
                ),

            "R2":
                round(
                    analysis[
                        "model_results"
                    ]["lasso"]["r2"],
                    6
                ),

            "non_zero":
                analysis[
                    "model_results"
                ]["lasso"][
                    "non_zero_coefficients"
                ]
        }
    ]

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model_key = (
        analysis[
            "best_model_key"
        ]
    )

    best_model_name = (
        analysis[
            "best_model_name"
        ]
    )

    best_model_metrics = (
        analysis[
            "model_results"
        ][
            best_model_key
        ]
    )

    best_model_information = {

        "key":
            best_model_key,

        "name":
            best_model_name,

        "MAE":
            round(
                best_model_metrics[
                    "mae"
                ],
                6
            ),

        "MSE":
            round(
                best_model_metrics[
                    "mse"
                ],
                6
            ),

        "RMSE":
            round(
                best_model_metrics[
                    "rmse"
                ],
                6
            ),

        "R2":
            round(
                best_model_metrics[
                    "r2"
                ],
                6
            )
    }

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    selected_rmse = (
        selected_result[
            "rmse"
        ]
    )

    selected_r2 = (
        selected_result[
            "r2"
        ]
    )

    if model_type == "linear":

        selected_description = (
            "Linear Regression is the selected baseline model "
            "without regularization."
        )

    elif model_type == "ridge":

        selected_description = (
            "Ridge Regression is the selected model and "
            "uses L2 regularization to reduce the effect "
            "of large coefficients."
        )

    else:

        selected_description = (
            "Lasso Regression is the selected model and "
            "uses L1 regularization, which can reduce "
            "some feature coefficients to zero."
        )

    if best_model_key == model_type:

        best_statement = (
            "The selected model is also the best-performing "
            "model among Linear, Ridge, and Lasso based "
            "on the lowest RMSE."
        )

    else:

        best_statement = (

            "The best-performing model among the three "
            "models is "
            + best_model_name
            + ", based on the lowest RMSE."
        )

    conclusion = (

        selected_description
        + " The selected model achieved an RMSE of "
        + str(
            round(
                selected_rmse,
                4
            )
        )
        + " and an R² score of "
        + str(
            round(
                selected_r2,
                4
            )
        )
        + ". "
        + best_statement
        + " Lower MAE, MSE and RMSE indicate smaller "
        "prediction errors, while a higher R² indicates "
        "that a larger proportion of PM2.5 variation "
        "is explained by the model."
    )

    # --------------------------------------------------------
    # SAVE SELECTED PREDICTIONS
    # --------------------------------------------------------

    save_selected_predictions(
        actual,
        predicted,
        analysis[
            "test_metadata"
        ],
        model_type
    )

    # --------------------------------------------------------
    # SAVE SELECTED COEFFICIENTS
    # --------------------------------------------------------

    coefficient_save_df = (
        analysis[
            "model_results"
        ][
            model_type
        ][
            "coefficients"
        ]
    )

    coefficient_path = os.path.join(
        MODEL_DIR,
        "coefficients_"
        + model_type
        + ".csv"
    )

    coefficient_save_df.to_csv(
        coefficient_path,
        index=False
    )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    model_configuration = (
        analysis[
            "model_configurations"
        ][
            model_type
        ]
    ).copy()

    model_configuration[
        "Scaling"
    ] = "StandardScaler"

    model_configuration[
        "Train/Test Split"
    ] = (
        "Train: before 2017 | "
        "Test: 2017"
    )

    model_configuration[
        "Target"
    ] = "PM2.5"

    model_configuration[
        "Features Before Encoding"
    ] = len(
        analysis[
            "feature_target_information"
        ]["Feature list"]
    )

    model_configuration[
        "Features After Encoding"
    ] = len(
        analysis[
            "feature_names"
        ]
    )

    # --------------------------------------------------------
    # RETURN EVERYTHING FOR HTML
    # --------------------------------------------------------

    result = {

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        "problem_definition": {

            "title":
                "Air Quality PM2.5 Prediction",

            "objective":
                "Predict PM2.5 concentration using "
                "pollutant, meteorological and time-based "
                "features.",

            "target":
                "PM2.5",

            "problem_type":
                "Supervised Regression",

            "reason":
                "PM2.5 is a continuous numerical value, "
                "therefore regression is appropriate."
        },

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        "selected_model_key":
            model_type,

        "selected_model_name":
            selected_model_name,

        "available_models": [

            {
                "key": "linear",
                "name": "Linear Regression"
            },

            {
                "key": "ridge",
                "name": "Ridge Regression"
            },

            {
                "key": "lasso",
                "name": "Lasso Regression"
            }
        ],

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        "dataset_information":
            analysis[
                "dataset_information"
            ],

        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        "feature_target_information":
            analysis[
                "feature_target_information"
            ],

        # ----------------------------------------------------
        # STEP 5
        # ----------------------------------------------------

        "preprocessing_summary":
            analysis[
                "preprocessing_summary"
            ],

        # ----------------------------------------------------
        # STEP 6
        # ----------------------------------------------------

        "model_configuration":
            model_configuration,

        # ----------------------------------------------------
        # STEP 7
        # ----------------------------------------------------

        "evaluation":
            evaluation,

        # ----------------------------------------------------
        # STEP 8
        # ----------------------------------------------------

        "prediction_table":
            prediction_table,

        # ----------------------------------------------------
        # STEP 9
        # ----------------------------------------------------

        "coefficient_table":
            coefficient_table,

        "intercept":
            round(
                selected_result[
                    "intercept"
                ],
                6
            ),

        # ----------------------------------------------------
        # STEP 10
        # ----------------------------------------------------

        "correlation_table":
            correlation_table,

        # ----------------------------------------------------
        # STEP 11
        # ----------------------------------------------------

        "error_analysis":
            error_analysis,

        # ----------------------------------------------------
        # STEP 12-15
        # ----------------------------------------------------

        "charts": {

            "actual_vs_predicted":
                actual_predicted_chart,

            "residual_analysis":
                residual_chart,

            "feature_coefficients":
                coefficient_chart,

            "correlation":
                correlation_chart
        },

        # ----------------------------------------------------
        # STEP 16
        # ----------------------------------------------------

        "model_comparison":
            model_comparison,

        # ----------------------------------------------------
        # STEP 17
        # ----------------------------------------------------

        "best_model":
            best_model_information,

        # ----------------------------------------------------
        # STEP 18
        # ----------------------------------------------------

        "regularization_comparison":
            regularization_comparison,

        # ----------------------------------------------------
        # STEP 19
        # ----------------------------------------------------

        "conclusion":
            conclusion
    }

    return result


# ============================================================
# TEST DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("AEROPURE - LINEAR REGRESSION TEST")
    print("=" * 80)

    result = run_regression_analysis(
        model_type="linear"
    )

    print("\nSelected Model:")
    print(
        result[
            "selected_model_name"
        ]
    )

    print("\nEvaluation:")
    print(
        result[
            "evaluation"
        ]
    )

    print("\nBest Model:")
    print(
        result[
            "best_model"
        ]
    )

    print("\nConclusion:")
    print(
        result[
            "conclusion"
        ]
    )

    print("\n")
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)