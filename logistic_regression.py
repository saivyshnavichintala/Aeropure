import os
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    roc_curve
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "Merged_PRSA_Data.csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "logistic_regression_charts"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "model_results"
)

CACHE_FILE = os.path.join(
    RESULT_DIR,
    "logistic_regression_cache.joblib"
)

# Cache version
# Change this number whenever the result structure is changed.
CACHE_VERSION = 3

os.makedirs(
    CHART_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# MODEL NAMES
# ============================================================

MODEL_NAMES = {

    "logistic":
        "Logistic Regression (No Regularization)",

    "ridge":
        "Logistic Regression - Ridge (L2)",

    "lasso":
        "Logistic Regression - Lasso (L1)"
}


MODEL_KEYS = [
    "logistic",
    "ridge",
    "lasso"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    print(
        f"Dataset loaded successfully: "
        f"{len(df):,} rows, "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    # PM2.5 threshold
    threshold = 35.0

    # Remove rows where PM2.5 is missing
    df = df.dropna(
        subset=["PM2.5"]
    ).copy()

    # Binary classification
    #
    # 0 = Low PM2.5
    # 1 = High PM2.5

    df["PM2.5_Class"] = (
        df["PM2.5"] >= threshold
    ).astype(int)

    return df, threshold


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Remove duplicate uppercase Station column
    # --------------------------------------------------------

    if "Station" in df.columns:

        df = df.drop(
            columns=["Station"]
        )

    # --------------------------------------------------------
    # Create datetime
    #
    # IMPORTANT:
    # Do NOT create datetime using string joining.
    # Dataset columns may contain values such as 2017.0.
    # --------------------------------------------------------

    required_columns = [
        "year",
        "month",
        "day",
        "hour"
    ]

    if all(
        column in df.columns
        for column in required_columns
    ):

        year_values = pd.to_numeric(
            df["year"],
            errors="coerce"
        )

        month_values = pd.to_numeric(
            df["month"],
            errors="coerce"
        )

        day_values = pd.to_numeric(
            df["day"],
            errors="coerce"
        )

        hour_values = pd.to_numeric(
            df["hour"],
            errors="coerce"
        )

        df["datetime"] = pd.to_datetime(
            {
                "year": year_values,
                "month": month_values,
                "day": day_values,
                "hour": hour_values
            },
            errors="coerce"
        )

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------

        df["day_of_week"] = (
            df["datetime"].dt.dayofweek
        )

        df["day_of_year"] = (
            df["datetime"].dt.dayofyear
        )

        df["week_of_year"] = (
            df["datetime"]
            .dt.isocalendar()
            .week
            .astype(float)
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        # ----------------------------------------------------
        # Season
        # ----------------------------------------------------

        def get_season(month):

            if pd.isna(month):

                return "Unknown"

            month = int(month)

            if month in [12, 1, 2]:

                return "Winter"

            elif month in [3, 4, 5]:

                return "Spring"

            elif month in [6, 7, 8]:

                return "Summer"

            else:

                return "Autumn"

        df["season"] = (
            df["month"].apply(
                get_season
            )
        )

    return df


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(model_key):

    # --------------------------------------------------------
    # Logistic Regression without regularization
    # --------------------------------------------------------

    if model_key == "logistic":

        return LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        )

    # --------------------------------------------------------
    # Ridge - L2
    # --------------------------------------------------------

    elif model_key == "ridge":

        return LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=42
        )

    # --------------------------------------------------------
    # Lasso - L1
    # --------------------------------------------------------

    elif model_key == "lasso":

        return LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=42
        )

    else:

        raise ValueError(
            f"Unknown model key: {model_key}"
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    numeric_features,
    categorical_features
):

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
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
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # ColumnTransformer
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
        ]
    )

    return preprocessor


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(
    X_train,
    X_test,
    numeric_features,
    categorical_features
):

    print(
        "Starting preprocessing..."
    )

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features
    )

    # --------------------------------------------------------
    # Fit only on training data
    # --------------------------------------------------------

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        "Training preprocessing completed."
    )

    # --------------------------------------------------------
    # Transform test data
    # --------------------------------------------------------

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "Testing preprocessing completed."
    )

    return (
        preprocessor,
        X_train_transformed,
        X_test_transformed
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_single_model(
    model_key,
    X_train_transformed,
    X_test_transformed,
    y_train,
    y_test
):

    print(
        f"Training {MODEL_NAMES[model_key]}..."
    )

    model = create_model(
        model_key
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    model.fit(
        X_train_transformed,
        y_train
    )

    print(
        f"{MODEL_NAMES[model_key]} training completed."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_transformed
    )

    probabilities = (
        model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    loss = log_loss(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    coefficients = (
        model.coef_[0]
    )

    absolute_coefficients = (
        np.abs(coefficients)
    )

    # --------------------------------------------------------
    # Regularization diagnostics
    # --------------------------------------------------------

    non_zero = int(
        np.sum(
            absolute_coefficients > 1e-10
        )
    )

    zero_coefficients = int(
        np.sum(
            absolute_coefficients <= 1e-10
        )
    )

    near_zero = int(
        np.sum(
            absolute_coefficients < 1e-4
        )
    )

    l1_norm = float(
        np.sum(
            absolute_coefficients
        )
    )

    l2_norm = float(
        np.sqrt(
            np.sum(
                coefficients ** 2
            )
        )
    )

    max_abs_coefficient = float(
        np.max(
            absolute_coefficients
        )
    )

    mean_abs_coefficient = float(
        np.mean(
            absolute_coefficients
        )
    )

    # --------------------------------------------------------
    # Number of iterations
    # --------------------------------------------------------

    try:

        n_iterations = int(
            np.max(
                model.n_iter_
            )
        )

    except Exception:

        n_iterations = 0

    # --------------------------------------------------------
    # Model settings
    # --------------------------------------------------------

    if model_key == "logistic":

        penalty = "None"
        regularization = "None"
        C_value = "N/A"

    elif model_key == "ridge":

        penalty = "L2"
        regularization = "Ridge"
        C_value = 1.0

    else:

        penalty = "L1"
        regularization = "Lasso"
        C_value = 1.0

    return {

        "model":
            model,

        "predictions":
            predictions,

        "probabilities":
            probabilities,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "log_loss":
            float(loss),

        "confusion_matrix":
            cm,

        "coefficients":
            coefficients,

        "non_zero":
            non_zero,

        "zero_coefficients":
            zero_coefficients,

        "near_zero":
            near_zero,

        "l1_norm":
            l1_norm,

        "l2_norm":
            l2_norm,

        "max_abs_coefficient":
            max_abs_coefficient,

        "mean_abs_coefficient":
            mean_abs_coefficient,

        "penalty":
            penalty,

        "regularization":
            regularization,

        "C":
            C_value,

        "n_iterations":
            n_iterations
    }


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_feature_names(
    preprocessor
):

    try:

        return list(
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        return []


# ============================================================
# CORRELATION TABLE
# ============================================================

def create_correlation_table(
    train_df,
    numeric_features
):

    correlation_rows = []

    for feature in numeric_features:

        if feature not in train_df.columns:

            continue

        if "PM2.5" not in train_df.columns:

            continue

        correlation = (
            train_df[
                [
                    feature,
                    "PM2.5"
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        if pd.isna(correlation):

            continue

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if correlation > 0:

            direction = "Positive"

        elif correlation < 0:

            direction = "Negative"

        else:

            direction = "Neutral"

        # ----------------------------------------------------
        # Strength
        # ----------------------------------------------------

        absolute = abs(
            correlation
        )

        if absolute >= 0.7:

            strength = "Strong"

        elif absolute >= 0.4:

            strength = "Moderate"

        elif absolute >= 0.2:

            strength = "Weak"

        else:

            strength = "Very Weak"

        correlation_rows.append(
            {

                "feature":
                    feature,

                "correlation":
                    round(
                        float(
                            correlation
                        ),
                        6
                    ),

                "direction":
                    direction,

                "strength":
                    strength
            }
        )

    correlation_rows.sort(
        key=lambda row:
        abs(
            row["correlation"]
        ),
        reverse=True
    )

    return correlation_rows


# ============================================================
# WEB CHART PATH
# ============================================================

def web_chart_path(
    filename
):

    return (
        "logistic_regression_charts/"
        + filename
    )


# ============================================================
# CONFUSION MATRIX CHART
# ============================================================

def create_confusion_matrix_chart(
    cm,
    model_name,
    model_key
):

    filename = (
        f"confusion_matrix_{model_key}.png"
    )

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=[
            "Low",
            "High"
        ],
        yticklabels=[
            "Low",
            "High"
        ]
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve(
    y_test,
    probabilities,
    model_name,
    model_key
):

    filename = (
        f"roc_curve_{model_key}.png"
    )

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC = {auc:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# FEATURE COEFFICIENT CHART
# ============================================================

def create_coefficient_chart(
    feature_names,
    coefficients,
    model_key
):

    filename = (
        f"feature_coefficients_{model_key}.png"
    )

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    coefficient_df = pd.DataFrame(
        {

            "feature":
                feature_names,

            "coefficient":
                coefficients
        }
    )

    coefficient_df[
        "absolute"
    ] = (
        coefficient_df[
            "coefficient"
        ].abs()
    )

    # Top 20
    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute",
            ascending=False
        )
        .head(20)
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            "coefficient"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        coefficient_df["feature"],
        coefficient_df["coefficient"]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Feature Coefficients"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# CORRELATION CHART
# ============================================================

def create_correlation_chart(
    correlation_table
):

    filename = "correlation.png"

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    correlation_df = pd.DataFrame(
        correlation_table
    )

    if correlation_df.empty:

        return ""

    correlation_df = (
        correlation_df
        .sort_values(
            "correlation"
        )
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        correlation_df["feature"],
        correlation_df["correlation"]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Pearson Correlation"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Correlation with PM2.5"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# FIND BEST MODEL
# ============================================================

def find_best_model(
    all_results
):

    # --------------------------------------------------------
    # Primary:
    # F1 score
    #
    # Tie breaker:
    # ROC-AUC
    #
    # Final tie breaker:
    # Lowest Log Loss
    # --------------------------------------------------------

    best_key = max(

        MODEL_KEYS,

        key=lambda key: (

            all_results[key]["f1"],

            all_results[key]["roc_auc"],

            -all_results[key]["log_loss"]
        )
    )

    return best_key


# ============================================================
# BUILD SELECTED RESULT
# ============================================================

def build_selected_result(
    model_type,
    all_results,
    feature_names,
    correlation_table,
    dataset_information,
    feature_target_information,
    preprocessing_summary,
    model_configuration_base,
    test_df,
    y_test,
    threshold
):

    selected = all_results[
        model_type
    ]

    selected_name = MODEL_NAMES[
        model_type
    ]

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_key = find_best_model(
        all_results
    )

    best_result = all_results[
        best_key
    ]

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluation = {

        "Accuracy":
            round(
                selected["accuracy"],
                6
            ),

        "Precision":
            round(
                selected["precision"],
                6
            ),

        "Recall":
            round(
                selected["recall"],
                6
            ),

        "F1 Score":
            round(
                selected["f1"],
                6
            ),

        "ROC-AUC":
            round(
                selected["roc_auc"],
                6
            ),

        "Log Loss":
            round(
                selected["log_loss"],
                6
            )
    }

    # ========================================================
    # REGULARIZATION VALUES
    # ========================================================

    regularization_values = {

        "Penalty":
            selected["penalty"],

        "Regularization":
            selected["regularization"],

        "C":
            selected["C"],

        "L1 Norm":
            round(
                selected["l1_norm"],
                6
            ),

        "L2 Norm":
            round(
                selected["l2_norm"],
                6
            ),

        "Maximum Absolute Coefficient":
            round(
                selected["max_abs_coefficient"],
                6
            ),

        "Average Absolute Coefficient":
            round(
                selected["mean_abs_coefficient"],
                6
            ),

        "Near-Zero Coefficients":
            selected["near_zero"],

        "Zero Coefficients":
            selected["zero_coefficients"],

        "Non-Zero Coefficients":
            selected["non_zero"],

        "Total Coefficients":
            len(
                selected["coefficients"]
            )
    }

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    prediction_table = []

    table_test = (
        test_df
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    if "datetime" in table_test.columns:

        datetime_series = pd.to_datetime(
            table_test["datetime"],
            errors="coerce"
        )

        datetime_values = (
            datetime_series
            .dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            .fillna("-")
            .tolist()
        )

    else:

        datetime_values = [
            "-"
            for _ in range(
                len(table_test)
            )
        ]

    # --------------------------------------------------------
    # Station
    # --------------------------------------------------------

    if "station" in table_test.columns:

        station_values = (
            table_test["station"]
            .fillna("-")
            .astype(str)
            .tolist()
        )

    else:

        station_values = [
            "-"
            for _ in range(
                len(table_test)
            )
        ]

    y_test_values = np.asarray(
        y_test
    )

    predictions = selected[
        "predictions"
    ]

    probabilities = selected[
        "probabilities"
    ]

    number_of_rows = min(
        50,
        len(y_test_values),
        len(predictions),
        len(probabilities)
    )

    for i in range(
        number_of_rows
    ):

        actual = int(
            y_test_values[i]
        )

        predicted = int(
            predictions[i]
        )

        probability = float(
            probabilities[i]
        )

        prediction_table.append(
            {

                "datetime":
                    datetime_values[i],

                "station":
                    station_values[i],

                "actual":
                    (
                        "High"
                        if actual == 1
                        else "Low"
                    ),

                "predicted":
                    (
                        "High"
                        if predicted == 1
                        else "Low"
                    ),

                "probability":
                    round(
                        probability,
                        6
                    ),

                "error":
                    actual - predicted,

                "correct":
                    (
                        "Yes"
                        if actual == predicted
                        else "No"
                    )
            }
        )

    # ========================================================
    # COEFFICIENT TABLE
    # ========================================================

    coefficient_table = []

    for feature, coefficient in zip(
        feature_names,
        selected["coefficients"]
    ):

        coefficient = float(
            coefficient
        )

        coefficient_table.append(
            {

                "feature":
                    feature,

                "coefficient":
                    round(
                        coefficient,
                        6
                    ),

                "absolute_coefficient":
                    round(
                        abs(coefficient),
                        6
                    ),

                "direction":
                    (
                        "Positive"
                        if coefficient > 0

                        else "Negative"
                        if coefficient < 0

                        else "Zero"
                    )
            }
        )

    coefficient_table.sort(
        key=lambda row:
        row["absolute_coefficient"],
        reverse=True
    )

    coefficient_table = (
        coefficient_table[:30]
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    cm = selected[
        "confusion_matrix"
    ]

    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    tp = int(cm[1][1])

    total = (
        tn + fp + fn + tp
    )

    incorrect = (
        fp + fn
    )

    error_percentage = (

        incorrect / total * 100

        if total > 0

        else 0
    )

    class_0_count = int(
        (
            y_test_values == 0
        ).sum()
    )

    class_1_count = int(
        (
            y_test_values == 1
        ).sum()
    )

    error_analysis = {

        "True Negatives":
            tn,

        "False Positives":
            fp,

        "False Negatives":
            fn,

        "True Positives":
            tp,

        "Correct Predictions":
            tn + tp,

        "Incorrect Predictions":
            incorrect,

        "Prediction Error Percentage":
            round(
                error_percentage,
                6
            ),

        "Class 0 Count":
            class_0_count,

        "Class 1 Count":
            class_1_count
    }

    # ========================================================
    # CHARTS
    # ========================================================

    charts = {

        "confusion_matrix":
            create_confusion_matrix_chart(
                selected["confusion_matrix"],
                selected_name,
                model_type
            ),

        "roc_curve":
            create_roc_curve(
                y_test,
                selected["probabilities"],
                selected_name,
                model_type
            ),

        "feature_coefficients":
            create_coefficient_chart(
                feature_names,
                selected["coefficients"],
                model_type
            ),

        "correlation":
            create_correlation_chart(
                correlation_table
            )
    }

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    model_comparison = []

    for key in MODEL_KEYS:

        result = all_results[
            key
        ]

        model_comparison.append(
            {

                "model":
                    MODEL_NAMES[key],

                "Accuracy":
                    round(
                        result["accuracy"],
                        6
                    ),

                "Precision":
                    round(
                        result["precision"],
                        6
                    ),

                "Recall":
                    round(
                        result["recall"],
                        6
                    ),

                "F1":
                    round(
                        result["f1"],
                        6
                    ),

                "ROC_AUC":
                    round(
                        result["roc_auc"],
                        6
                    ),

                "Log_Loss":
                    round(
                        result["log_loss"],
                        6
                    ),

                "non_zero_coefficients":
                    result["non_zero"],

                "zero_coefficients":
                    result["zero_coefficients"],

                "near_zero_coefficients":
                    result["near_zero"],

                "penalty":
                    result["penalty"],

                "C":
                    result["C"],

                "selected":
                    key == model_type,

                "best":
                    key == best_key
            }
        )

    # ========================================================
    # BEST MODEL INFORMATION
    # ========================================================

    best_model = {

        "key":
            best_key,

        "name":
            MODEL_NAMES[best_key],

        "Accuracy":
            round(
                best_result["accuracy"],
                6
            ),

        "Precision":
            round(
                best_result["precision"],
                6
            ),

        "Recall":
            round(
                best_result["recall"],
                6
            ),

        "F1":
            round(
                best_result["f1"],
                6
            ),

        "ROC_AUC":
            round(
                best_result["roc_auc"],
                6
            ),

        "Log_Loss":
            round(
                best_result["log_loss"],
                6
            )
    }

    # ========================================================
    # REGULARIZATION COMPARISON
    # ========================================================

    regularization_comparison = []

    for key in MODEL_KEYS:

        result = all_results[
            key
        ]

        regularization_comparison.append(
            {

                "model":
                    MODEL_NAMES[key],

                "regularization":
                    result["regularization"],

                "penalty":
                    result["penalty"],

                "C":
                    result["C"],

                "F1":
                    round(
                        result["f1"],
                        6
                    ),

                "ROC_AUC":
                    round(
                        result["roc_auc"],
                        6
                    ),

                "Log_Loss":
                    round(
                        result["log_loss"],
                        6
                    ),

                "non_zero":
                    result["non_zero"],

                "zero_coefficients":
                    result["zero_coefficients"],

                "near_zero":
                    result["near_zero"],

                "l1_norm":
                    round(
                        result["l1_norm"],
                        6
                    ),

                "l2_norm":
                    round(
                        result["l2_norm"],
                        6
                    )
            }
        )

    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    model_configuration = dict(
        model_configuration_base
    )

    model_configuration.update(
        {

            "Selected Model":
                selected_name,

            "Selected Penalty":
                selected["penalty"],

            "Selected C":
                selected["C"],

            "Selected Regularization":
                selected["regularization"],

            "Best Model":
                MODEL_NAMES[best_key],

            "Best Model Selection":
                (
                    "Highest F1 score, followed by "
                    "ROC-AUC and then lowest Log Loss"
                )
        }
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    conclusion = (

        f"Logistic Regression was used to classify "
        f"PM2.5 concentration into Low and High classes "
        f"using a threshold of {threshold} µg/m³. "

        f"The selected model, {selected_name}, achieved "
        f"an accuracy of {selected['accuracy']:.4f}, "
        f"precision of {selected['precision']:.4f}, "
        f"recall of {selected['recall']:.4f}, "
        f"F1 score of {selected['f1']:.4f}, "
        f"and ROC-AUC of {selected['roc_auc']:.4f}. "

        f"The three Logistic Regression variants were "
        f"trained using the same chronological training "
        f"and testing split. "

        f"L1 regularization applies an absolute-value "
        f"penalty and can produce sparse coefficients, "
        f"while L2 regularization applies a squared "
        f"penalty and generally shrinks coefficient "
        f"magnitudes without forcing them exactly to zero. "

        f"The best model based on F1 score, ROC-AUC "
        f"and Log Loss tie-breaking was "
        f"{MODEL_NAMES[best_key]}."
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "problem_definition": {

            "title":
                "PM2.5 Air Quality Classification",

            "objective":
                (
                    "Classify PM2.5 concentration as "
                    "Low or High using pollutant, "
                    "meteorological and time-based features."
                ),

            "target":
                "PM2.5_Class",

            "problem_type":
                "Supervised Binary Classification",

            "target_description":
                (
                    "0 represents Low PM2.5 and "
                    "1 represents High PM2.5."
                ),

            "why_logistic":
                (
                    "The target is converted into two "
                    "classes, making Logistic Regression "
                    "suitable."
                )
        },

        "selected_model_key":
            model_type,

        "selected_model_name":
            selected_name,

        "best_model":
            best_model,

        "dataset_information":
            dataset_information,

        "feature_target_information":
            feature_target_information,

        "preprocessing_summary":
            preprocessing_summary,

        "model_configuration":
            model_configuration,

        "evaluation":
            evaluation,

        "regularization_values":
            regularization_values,

        "prediction_table":
            prediction_table,

        "coefficient_table":
            coefficient_table,

        "intercept":
            round(
                float(
                    selected[
                        "model"
                    ].intercept_[0]
                ),
                6
            ),

        "correlation_table":
            correlation_table,

        "error_analysis":
            error_analysis,

        "charts":
            charts,

        "model_comparison":
            model_comparison,

        "regularization_comparison":
            regularization_comparison,

        "conclusion":
            conclusion
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_logistic_regression_analysis(
    model_type="logistic"
):

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if model_type not in MODEL_KEYS:

        model_type = "logistic"

    # ========================================================
    # TRY CACHE
    # ========================================================

    if os.path.exists(
        CACHE_FILE
    ):

        try:

            print(
                "Loading cached Logistic Regression results..."
            )

            cache = joblib.load(
                CACHE_FILE
            )

            # ------------------------------------------------
            # Check cache version
            # ------------------------------------------------

            if cache.get(
                "cache_version"
            ) != CACHE_VERSION:

                print(
                    "Old cache version detected."
                )

                print(
                    "Recalculating Logistic Regression..."
                )

                os.remove(
                    CACHE_FILE
                )

            else:

                required_keys = [

                    "all_results",
                    "feature_names",
                    "correlation_table",
                    "dataset_information",
                    "feature_target_information",
                    "preprocessing_summary",
                    "model_configuration_base",
                    "test_df",
                    "y_test",
                    "threshold"
                ]

                cache_valid = all(
                    key in cache
                    for key in required_keys
                )

                # ------------------------------------------------
                # Check all models
                # ------------------------------------------------

                if cache_valid:

                    cache_valid = all(
                        key in cache[
                            "all_results"
                        ]
                        for key in MODEL_KEYS
                    )

                if cache_valid:

                    print(
                        "Cached results loaded successfully."
                    )

                    return build_selected_result(

                        model_type,

                        cache[
                            "all_results"
                        ],

                        cache[
                            "feature_names"
                        ],

                        cache[
                            "correlation_table"
                        ],

                        cache[
                            "dataset_information"
                        ],

                        cache[
                            "feature_target_information"
                        ],

                        cache[
                            "preprocessing_summary"
                        ],

                        cache[
                            "model_configuration_base"
                        ],

                        cache[
                            "test_df"
                        ],

                        cache[
                            "y_test"
                        ],

                        cache[
                            "threshold"
                        ]
                    )

                else:

                    print(
                        "Incomplete cache detected."
                    )

                    os.remove(
                        CACHE_FILE
                    )

        except Exception as e:

            print(
                f"Cache error: {e}"
            )

            print(
                "Recalculating models..."
            )

            try:

                if os.path.exists(
                    CACHE_FILE
                ):

                    os.remove(
                        CACHE_FILE
                    )

            except Exception:

                pass

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "Training Logistic Regression models..."
    )

    df = load_dataset()

    original_rows = len(
        df
    )

    original_columns = len(
        df.columns
    )

    # ========================================================
    # CREATE TARGET
    # ========================================================

    df, threshold = create_target(
        df
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = prepare_features(
        df
    )

    # ========================================================
    # TARGET
    # ========================================================

    target = "PM2.5_Class"

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

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

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    categorical_features = [

        "wd",
        "season",
        "station"
    ]

    # --------------------------------------------------------
    # Keep only columns actually present
    # --------------------------------------------------------

    numeric_features = [

        column

        for column in numeric_features

        if column in df.columns
    ]

    categorical_features = [

        column

        for column in categorical_features

        if column in df.columns
    ]

    feature_columns = (
        numeric_features
        +
        categorical_features
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    train_df = df[
        df["year"] < 2017
    ].copy()

    test_df = df[
        df["year"] >= 2017
    ].copy()

    # ========================================================
    # X / Y
    #
    # IMPORTANT:
    # datetime is NOT included here.
    # It is only used for display.
    # ========================================================

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        target
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        target
    ].copy()

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    dataset_information = {

        "Dataset":
            "Beijing Multi-Site Air Quality Dataset",

        "Original Rows":
            f"{original_rows:,}",

        "Original Columns":
            original_columns,

        "Rows After Target Cleaning":
            f"{len(df):,}",

        "Training Rows":
            f"{len(train_df):,}",

        "Testing Rows":
            f"{len(test_df):,}",

        "Training Period":
            "2013-2016",

        "Testing Period":
            "2017",

        "Classification Threshold":
            f"{threshold} µg/m³",

        "Class 0":
            "Low PM2.5",

        "Class 1":
            "High PM2.5"
    }

    # ========================================================
    # FEATURE TARGET INFORMATION
    # ========================================================

    feature_target_information = {

        "Target":
            target,

        "Target description":
            (
                "Binary classification of "
                "PM2.5 concentration"
            ),

        "Classification rule":
            (
                f"PM2.5 < {threshold} = Low, "
                f"PM2.5 >= {threshold} = High"
            ),

        "Total original features":
            len(feature_columns),

        "Numeric feature count":
            len(numeric_features),

        "Categorical feature count":
            len(categorical_features),

        "Transformed feature count":
            "50 after one-hot encoding",

        "Feature list":
            feature_columns
    }

    # ========================================================
    # PREPROCESSING SUMMARY
    # ========================================================

    preprocessing_summary = {

        "Missing numeric values":
            "Median imputation",

        "Missing categorical values":
            "Most frequent value imputation",

        "Numeric scaling":
            "StandardScaler",

        "Categorical encoding":
            "One-Hot Encoding",

        "Unknown categories":
            "Ignored using handle_unknown='ignore'",

        "Data split":
            "Chronological split",

        "Training data":
            "2013-2016",

        "Testing data":
            "2017",

        "Data leakage prevention":
            (
                "Preprocessor fitted only on "
                "training data"
            )
    }

    # ========================================================
    # PREPROCESS
    # ========================================================

    (
        preprocessor,
        X_train_transformed,
        X_test_transformed
    ) = preprocess_data(

        X_train,

        X_test,

        numeric_features,

        categorical_features
    )

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = get_feature_names(
        preprocessor
    )

    print(
        f"Transformed feature count: "
        f"{len(feature_names)}"
    )

    # ========================================================
    # TRAIN ALL THREE MODELS
    # ========================================================

    all_results = {}

    for key in MODEL_KEYS:

        all_results[key] = (
            train_single_model(

                key,

                X_train_transformed,

                X_test_transformed,

                y_train,

                y_test
            )
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOGISTIC REGRESSION MODEL COMPARISON")
    print("=" * 70)

    for key in MODEL_KEYS:

        result = all_results[key]

        print(
            f"\n{MODEL_NAMES[key]}"
        )

        print(
            f"Penalty: {result['penalty']}"
        )

        print(
            f"C: {result['C']}"
        )

        print(
            f"Accuracy: {result['accuracy']:.6f}"
        )

        print(
            f"Precision: {result['precision']:.6f}"
        )

        print(
            f"Recall: {result['recall']:.6f}"
        )

        print(
            f"F1: {result['f1']:.6f}"
        )

        print(
            f"ROC-AUC: {result['roc_auc']:.6f}"
        )

        print(
            f"Log Loss: {result['log_loss']:.6f}"
        )

        print(
            f"Non-zero coefficients: "
            f"{result['non_zero']}"
        )

        print(
            f"Zero coefficients: "
            f"{result['zero_coefficients']}"
        )

        print(
            f"Near-zero coefficients: "
            f"{result['near_zero']}"
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    correlation_table = (
        create_correlation_table(
            train_df,
            numeric_features
        )
    )

    # ========================================================
    # BASE CONFIGURATION
    # ========================================================

    model_configuration_base = {

        "Algorithm":
            "Logistic Regression",

        "Solver":
            (
                "lbfgs for No Regularization, "
                "liblinear for L1/L2"
            ),

        "Maximum Iterations":
            2000,

        "No Regularization Model":
            "No penalty",

        "Ridge Model":
            "L2 penalty",

        "Lasso Model":
            "L1 penalty",

        "C for Ridge":
            1.0,

        "C for Lasso":
            1.0,

        "Classification Threshold":
            "0.50 probability",

        "L1 Testing":
            (
                "Compare coefficient sparsity, "
                "near-zero coefficients and L1 norm"
            ),

        "L2 Testing":
            (
                "Compare coefficient shrinkage, "
                "L2 norm and coefficient magnitude"
            )
    }

    # ========================================================
    # SAVE CACHE
    # ========================================================

    cache = {

        "cache_version":
            CACHE_VERSION,

        "all_results":
            all_results,

        "feature_names":
            feature_names,

        "correlation_table":
            correlation_table,

        "dataset_information":
            dataset_information,

        "feature_target_information":
            feature_target_information,

        "preprocessing_summary":
            preprocessing_summary,

        "model_configuration_base":
            model_configuration_base,

        # ----------------------------------------------------
        # Datetime is stored ONLY for display
        # ----------------------------------------------------

        "test_df":
            test_df[
                [
                    column
                    for column in [
                        "datetime",
                        "station",
                        target
                    ]
                    if column in test_df.columns
                ]
            ].copy(),

        "y_test":
            np.asarray(
                y_test
            ),

        "threshold":
            threshold
    }

    try:

        joblib.dump(
            cache,
            CACHE_FILE
        )

        print(
            "Logistic Regression cache saved successfully."
        )

    except Exception as e:

        print(
            f"Could not save cache: {e}"
        )

    # ========================================================
    # RETURN SELECTED MODEL
    # ========================================================

    return build_selected_result(

        model_type,

        all_results,

        feature_names,

        correlation_table,

        dataset_information,

        feature_target_information,

        preprocessing_summary,

        model_configuration_base,

        test_df,

        y_test,

        threshold
    )