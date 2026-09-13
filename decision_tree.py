import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "Merged_PRSA_Data.csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "decision_tree_charts"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "model_results"
)

CACHE_FILE = os.path.join(
    RESULT_DIR,
    "decision_tree_cache.joblib"
)

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Use limited rows because the original dataset is very large.
MAX_ROWS = 20000

PM25_THRESHOLD = 35.0

CACHE_VERSION = 6


# ============================================================
# DECISION TREE CALCULATIONS
# ============================================================

def calculate_entropy(y):
    """
    Entropy:
        H(S) = -Σ p_i log2(p_i)
    """

    y = pd.Series(y)

    probabilities = y.value_counts(normalize=True)

    entropy = 0.0

    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def calculate_gini(y):
    """
    Gini Impurity:
        Gini = 1 - Σ p_i²
    """

    y = pd.Series(y)

    probabilities = y.value_counts(normalize=True)

    gini = 1.0 - np.sum(probabilities ** 2)

    return float(gini)


def calculate_information_gain(parent, children):
    """
    Information Gain:

        IG =
        Entropy(parent)
        -
        weighted entropy(children)
    """

    parent = pd.Series(parent)

    parent_entropy = calculate_entropy(parent)

    total = len(parent)

    weighted_child_entropy = 0.0

    for child in children:

        child = pd.Series(child)

        if len(child) == 0:
            continue

        weight = len(child) / total

        weighted_child_entropy += (
            weight * calculate_entropy(child)
        )

    information_gain = (
        parent_entropy -
        weighted_child_entropy
    )

    return float(information_gain)


def calculate_split_information(children):
    """
    Split Information:

        SplitInfo =
        -Σ (|S_i| / |S|) log2(|S_i| / |S|)
    """

    total = sum(len(child) for child in children)

    split_info = 0.0

    for child in children:

        if len(child) == 0:
            continue

        proportion = len(child) / total

        if proportion > 0:
            split_info -= (
                proportion *
                np.log2(proportion)
            )

    return float(split_info)


def calculate_gain_ratio(parent, children):
    """
    Gain Ratio:

        Gain Ratio =
        Information Gain / Split Information
    """

    information_gain = calculate_information_gain(
        parent,
        children
    )

    split_information = calculate_split_information(
        children
    )

    if split_information == 0:
        return 0.0

    return float(
        information_gain /
        split_information
    )


# ============================================================
# SAMPLE CALCULATION FOR WEB PAGE
# ============================================================

def get_concept_calculations():

    # Example dataset:
    #
    # 10 samples
    # 6 = High
    # 4 = Low

    example = pd.Series(
        [
            "High",
            "High",
            "High",
            "High",
            "High",
            "High",
            "Low",
            "Low",
            "Low",
            "Low"
        ]
    )

    high = example[example == "High"]

    low = example[example == "Low"]

    entropy_parent = calculate_entropy(example)

    entropy_high = calculate_entropy(high)

    entropy_low = calculate_entropy(low)

    information_gain = calculate_information_gain(
        example,
        [high, low]
    )

    gini = calculate_gini(example)

    split_information = calculate_split_information(
        [high, low]
    )

    gain_ratio = calculate_gain_ratio(
        example,
        [high, low]
    )

    return {

        "example_total": len(example),

        "high_count": len(high),

        "low_count": len(low),

        "high_probability":
            round(len(high) / len(example), 4),

        "low_probability":
            round(len(low) / len(example), 4),

        "entropy":
            round(entropy_parent, 6),

        "high_entropy":
            round(entropy_high, 6),

        "low_entropy":
            round(entropy_low, 6),

        "information_gain":
            round(information_gain, 6),

        "gini":
            round(gini, 6),

        "split_information":
            round(split_information, 6),

        "gain_ratio":
            round(gain_ratio, 6)
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_decision_tree_data():

    print("Loading Decision Tree data...")

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    print(
        f"Original dataset: "
        f"{len(df):,} rows x {len(df.columns)} columns"
    )

    # Remove duplicate uppercase Station column
    if "Station" in df.columns:
        df = df.drop(columns=["Station"])

    # Convert PM2.5
    df["PM2.5"] = pd.to_numeric(
        df["PM2.5"],
        errors="coerce"
    )

    # Create classification target
    df["PM2.5_Class"] = np.where(
        df["PM2.5"] >= PM25_THRESHOLD,
        1,
        0
    )

    # Remove rows without target
    df = df.dropna(
        subset=["PM2.5_Class"]
    )

    # ========================================================
    # DATETIME
    # ========================================================

    df["datetime"] = pd.to_datetime(
        {
            "year": pd.to_numeric(
                df["year"],
                errors="coerce"
            ),
            "month": pd.to_numeric(
                df["month"],
                errors="coerce"
            ),
            "day": pd.to_numeric(
                df["day"],
                errors="coerce"
            ),
            "hour": pd.to_numeric(
                df["hour"],
                errors="coerce"
            )
        },
        errors="coerce"
    )

    # ========================================================
    # TIME FEATURES
    # ========================================================

    df["day_of_week"] = (
        df["datetime"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["datetime"].dt.dayofyear
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ========================================================
    # SEASON
    # ========================================================

    def get_season(month):

        if month in [3, 4, 5]:
            return "Spring"

        if month in [6, 7, 8]:
            return "Summer"

        if month in [9, 10, 11]:
            return "Autumn"

        return "Winter"

    df["season"] = df["month"].apply(
        get_season
    )

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

    # Keep only available columns
    numeric_features = [
        col for col in numeric_features
        if col in df.columns
    ]

    categorical_features = [
        col for col in categorical_features
        if col in df.columns
    ]

    features = (
        numeric_features +
        categorical_features
    )

    # ========================================================
    # SAMPLE DATA
    # ========================================================

    if len(df) > MAX_ROWS:

        print(
            f"Sampling {MAX_ROWS:,} rows "
            f"for faster Decision Tree training..."
        )

        sampled_df, _ = train_test_split(
            df,
            test_size=len(df) - MAX_ROWS,
            stratify=df["PM2.5_Class"],
            random_state=RANDOM_STATE
        )

        df = sampled_df.copy().reset_index(
            drop=True
        )

    print(
        f"Training dataset after sampling: "
        f"{len(df):,} rows"
    )

    # ========================================================
    # X AND y
    # ========================================================

    X = df[features].copy()

    y = df["PM2.5_Class"].astype(int)

    # ========================================================
    # NUMERIC CLEANING
    # ========================================================

    for col in numeric_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        X[col] = X[col].fillna(
            X[col].median()
        )

    # ========================================================
    # CATEGORICAL CLEANING
    # ========================================================

    for col in categorical_features:

        X[col] = X[col].astype(str)

        X[col] = X[col].replace(
            "nan",
            "Unknown"
        )

        X[col] = X[col].fillna(
            "Unknown"
        )

    # ========================================================
    # ONE HOT ENCODING
    # ========================================================

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=False
    )

    X_encoded = X_encoded.astype(float)

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows: {len(X_test):,}"
    )

    print(
        f"Features after encoding: "
        f"{X_encoded.shape[1]}"
    )

    return {

        "df": df,

        "X": X_encoded,

        "y": y,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "features": features,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features
    }


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_type):

    # ========================================================
    # ID3
    # ========================================================

    if model_type == "id3":

        return DecisionTreeClassifier(
            criterion="entropy",
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # C4.5
    # ========================================================

    elif model_type == "c45":

        # sklearn does not implement literal C4.5.
        # Entropy-based tree is used as a practical
        # approximation.

        return DecisionTreeClassifier(
            criterion="entropy",
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # CART
    # ========================================================

    elif model_type == "cart":

        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # PRUNED CART
    # ========================================================

    elif model_type == "pruned":

        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=40,
            min_samples_leaf=20,
            ccp_alpha=0.001,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    elif model_type == "random_forest":

        return RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # ADABOOST
    # ========================================================

    elif model_type == "adaboost":

        return AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.7,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # GRADIENT BOOSTING
    # ========================================================

    elif model_type == "gradient_boosting":

        return GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.08,
            max_depth=3,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )

    # ========================================================
    # XGBOOST
    # ========================================================

    elif model_type == "xgboost":

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. "
                "Run: pip install xgboost"
            )

        return XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )

    # ========================================================
    # LIGHTGBM
    # ========================================================

    elif model_type == "lightgbm":

        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. "
                "Run: pip install lightgbm"
            )

        return LGBMClassifier(
            n_estimators=50,
            learning_rate=0.08,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=-1,
            random_state=RANDOM_STATE
        )

    else:

        raise ValueError(
            f"Unknown model type: {model_type}"
        )


# ============================================================
# MODEL DESCRIPTIONS
# ============================================================

def get_model_description(model_type):

    descriptions = {

        "id3":
            "ID3 uses Entropy and Information Gain "
            "to select the best splitting attribute.",

        "c45":
            "C4.5 extends ID3 and commonly uses "
            "Gain Ratio for selecting splits. "
            "This implementation uses an entropy-based "
            "Decision Tree as a practical approximation.",

        "cart":
            "CART stands for Classification and Regression "
            "Trees. For classification it commonly uses "
            "Gini impurity and creates binary splits.",

        "pruned":
            "Pruned CART controls tree complexity using "
            "maximum depth, minimum samples and "
            "cost-complexity pruning.",

        "random_forest":
            "Random Forest combines many independent "
            "decision trees using bagging and random "
            "feature selection.",

        "adaboost":
            "AdaBoost sequentially builds weak learners "
            "and gives more importance to incorrectly "
            "classified observations.",

        "gradient_boosting":
            "Gradient Boosting sequentially builds trees "
            "that reduce the errors of previous trees.",

        "xgboost":
            "XGBoost is an optimized gradient boosting "
            "algorithm using regularization, shrinkage "
            "and efficient tree construction.",

        "lightgbm":
            "LightGBM is a fast gradient boosting framework "
            "using histogram-based learning and leaf-wise "
            "tree growth."
    }

    return descriptions.get(
        model_type,
        ""
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_selected_model(data, model_type):

    print(
        "\n================================="
    )

    print(
        f"Training: {model_type.upper()}"
    )

    print(
        "================================="
    )

    model = create_model(
        model_type
    )

    print(
        f"Training {model.__class__.__name__}..."
    )

    model.fit(
        data["X_train"],
        data["y_train"]
    )

    predictions = model.predict(
        data["X_test"]
    )

    probabilities = model.predict_proba(
        data["X_test"]
    )[:, 1]

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        data["y_test"],
        predictions
    )

    precision = precision_score(
        data["y_test"],
        predictions,
        zero_division=0
    )

    recall = recall_score(
        data["y_test"],
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        data["y_test"],
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        data["y_test"],
        probabilities
    )

    loss = log_loss(
        data["y_test"],
        probabilities
    )

    cm = confusion_matrix(
        data["y_test"],
        predictions
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    tn, fp, fn, tp = cm.ravel()

    correct = int(
        tn + tp
    )

    incorrect = int(
        fp + fn
    )

    error_percentage = (
        incorrect /
        len(data["y_test"])
    ) * 100

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = None

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = model.feature_importances_

        feature_importance = pd.DataFrame(
            {
                "feature":
                    data["X_train"].columns,

                "importance":
                    importance
            }
        )

        feature_importance = (
            feature_importance
            .sort_values(
                "importance",
                ascending=False
            )
            .head(20)
            .reset_index(drop=True)
        )

    # ========================================================
    # TREE INFORMATION
    # ========================================================

    tree_info = {}

    if hasattr(model, "tree_"):

        tree_info = {

            "node_count":
                int(model.tree_.node_count),

            "max_depth":
                int(model.tree_.max_depth),

            "leaf_count":
                int(
                    np.sum(
                        model.tree_.children_left == -1
                    )
                )
        }

    elif hasattr(model, "estimators_"):

        estimators = model.estimators_

        depths = []

        for estimator in estimators:

            if hasattr(estimator, "tree_"):

                depths.append(
                    estimator.tree_.max_depth
                )

        tree_info = {

            "number_of_trees":
                len(estimators),

            "average_depth":
                round(
                    np.mean(depths),
                    2
                ) if depths else 0,

            "maximum_depth":
                max(depths)
                if depths
                else 0
        }

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "model":
            model,

        "model_type":
            model_type,

        "model_name":
            model_type.replace(
                "_",
                " "
            ).title(),

        "description":
            get_model_description(
                model_type
            ),

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
            cm.tolist(),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "correct":
            correct,

        "incorrect":
            incorrect,

        "error_percentage":
            float(error_percentage),

        "feature_importance":
            feature_importance,

        "tree_information":
            tree_info,

        "predictions":
            predictions,

        "probabilities":
            probabilities
    }


# ============================================================
# CHART HELPERS
# ============================================================

def web_chart_path(filename):

    return (
        "decision_tree_charts/"
        + filename
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_chart(
    cm,
    model_type
):

    filename = (
        f"confusion_matrix_{model_type}.png"
    )

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(
        figsize=(6, 5)
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

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.title(
        f"Confusion Matrix - "
        f"{model_type.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

def create_feature_importance_chart(
    feature_importance,
    model_type
):

    filename = (
        f"feature_importance_{model_type}.png"
    )

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(
        figsize=(10, 7)
    )

    data = feature_importance.sort_values(
        "importance"
    )

    plt.barh(
        data["feature"],
        data["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Feature Importance - "
        f"{model_type.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# TREE STRUCTURE CHART
# ============================================================

def create_tree_chart(
    model,
    model_type,
    feature_names
):

    filename = (
        f"tree_structure_{model_type}.png"
    )

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    # Only actual single Decision Tree
    if not hasattr(model, "tree_"):

        return None

    plt.figure(
        figsize=(22, 12)
    )

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[
            "Low",
            "High"
        ],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8
    )

    plt.title(
        f"Tree Structure - "
        f"{model_type.upper()}"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=120
    )

    plt.close()

    return web_chart_path(
        filename
    )


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models(data):

    model_types = [

        "id3",

        "c45",

        "cart",

        "pruned",

        "random_forest",

        "adaboost",

        "gradient_boosting"
    ]

    if XGBOOST_AVAILABLE:
        model_types.append(
            "xgboost"
        )

    if LIGHTGBM_AVAILABLE:
        model_types.append(
            "lightgbm"
        )

    results = {}

    for model_type in model_types:

        try:

            results[model_type] = (
                train_selected_model(
                    data,
                    model_type
                )
            )

        except Exception as e:

            print(
                f"Error training "
                f"{model_type}: {e}"
            )

    return results


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_decision_tree_analysis(
    model_type="cart"
):

    print(
        "\nStarting Decision Tree Analysis..."
    )

    # ========================================================
    # CACHE
    # ========================================================

    if os.path.exists(CACHE_FILE):

        try:

            cached = joblib.load(
                CACHE_FILE
            )

            if (
                cached.get(
                    "cache_version"
                )
                == CACHE_VERSION
            ):

                print(
                    "Loading Decision Tree "
                    "results from cache..."
                )

                all_results = cached[
                    "all_results"
                ]

                data = cached[
                    "data"
                ]

                concept_calculations = cached[
                    "concept_calculations"
                ]

            else:

                print(
                    "Old cache detected. "
                    "Recalculating..."
                )

                raise ValueError(
                    "Old cache"
                )

        except Exception:

            data = load_decision_tree_data()

            all_results = train_all_models(
                data
            )

            concept_calculations = (
                get_concept_calculations()
            )

    else:

        data = load_decision_tree_data()

        all_results = train_all_models(
            data
        )

        concept_calculations = (
            get_concept_calculations()
        )

    # ========================================================
    # VALID MODEL
    # ========================================================

    if model_type not in all_results:

        model_type = "cart"

        if model_type not in all_results:

            model_type = next(
                iter(all_results)
            )

    selected = all_results[
        model_type
    ]

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    model_comparison = []

    for key, result in all_results.items():

        model_comparison.append(
            {

                "model":
                    result["model_name"],

                "accuracy":
                    result["accuracy"],

                "precision":
                    result["precision"],

                "recall":
                    result["recall"],

                "f1":
                    result["f1"],

                "roc_auc":
                    result["roc_auc"],

                "log_loss":
                    result["log_loss"],

                "selected":
                    key == model_type,

                "status":
                    ""
            }
        )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_key = max(
        all_results.keys(),
        key=lambda key: (
            all_results[key]["f1"],
            all_results[key]["roc_auc"],
            -all_results[key]["log_loss"]
        )
    )

    all_results[best_key][
        "best"
    ] = True

    for row in model_comparison:

        if (
            row["model"]
            == all_results[best_key][
                "model_name"
            ]
        ):

            row["status"] = "Best"

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        selected["feature_importance"]
    )

    feature_importance_records = []

    if feature_importance is not None:

        feature_importance_records = (
            feature_importance
            .to_dict(
                orient="records"
            )
        )

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    test_df = data["df"].loc[
        data["X_test"].index
    ].copy()

    test_df["Actual"] = (
        data["y_test"].values
    )

    test_df["Predicted"] = (
        selected["predictions"]
    )

    test_df["Probability"] = (
        selected["probabilities"]
    )

    test_df["Actual_Label"] = (
        test_df["Actual"]
        .map(
            {
                0: "Low",
                1: "High"
            }
        )
    )

    test_df["Predicted_Label"] = (
        test_df["Predicted"]
        .map(
            {
                0: "Low",
                1: "High"
            }
        )
    )

    # First 100 records for webpage
    prediction_table = []

    for _, row in test_df.head(
        100
    ).iterrows():

        prediction_table.append(
            {

                "datetime":
                    pd.to_datetime(
                        row.get(
                            "datetime"
                        ),
                        errors="coerce"
                    ).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if pd.notna(
                        row.get(
                            "datetime"
                        )
                    )
                    else "-",

                "actual":
                    int(row["Actual"]),

                "actual_label":
                    row["Actual_Label"],

                "predicted":
                    int(row["Predicted"]),

                "predicted_label":
                    row["Predicted_Label"],

                "probability":
                    round(
                        float(
                            row["Probability"]
                        ),
                        4
                    ),

                "correct":
                    int(
                        row["Actual"]
                        ==
                        row["Predicted"]
                    )
            }
        )

    # ========================================================
    # CHARTS
    # ========================================================

    charts = {}

    charts[
        "confusion_matrix"
    ] = create_confusion_matrix_chart(
        np.array(
            selected["confusion_matrix"]
        ),
        model_type
    )

    if feature_importance is not None:

        charts[
            "feature_importance"
        ] = create_feature_importance_chart(
            feature_importance,
            model_type
        )

    charts[
        "tree_structure"
    ] = create_tree_chart(
        selected["model"],
        model_type,
        list(
            data["X_train"].columns
        )
    )

    # ========================================================
    # REGRESSION TREE INFORMATION
    # ========================================================

    regression_tree_info = {

        "description":
            "Regression Trees predict a continuous "
            "numeric value instead of a class.",

        "split_criterion":
            "Mean Squared Error (MSE)",

        "leaf_prediction":
            "Mean value of target samples in the leaf",

        "metrics":
            "MSE, RMSE, MAE and R²"
    }

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "selected_model": {

            "key":
                model_type,

            "name":
                selected["model_name"],

            "description":
                selected["description"],

            "accuracy":
                selected["accuracy"],

            "precision":
                selected["precision"],

            "recall":
                selected["recall"],

            "f1":
                selected["f1"],

            "roc_auc":
                selected["roc_auc"],

            "log_loss":
                selected["log_loss"],

            "confusion_matrix":
                selected[
                    "confusion_matrix"
                ],

            "TN":
                selected["TN"],

            "FP":
                selected["FP"],

            "FN":
                selected["FN"],

            "TP":
                selected["TP"],

            "correct":
                selected["correct"],

            "incorrect":
                selected["incorrect"],

            "error_percentage":
                selected[
                    "error_percentage"
                ],

            "tree_information":
                selected[
                    "tree_information"
                ]
        },

        "dataset": {

            "original_rows":
                841536,

            "sampled_rows":
                len(data["df"]),

            "training_rows":
                len(data["X_train"]),

            "testing_rows":
                len(data["X_test"]),

            "original_columns":
                19,

            "features_before_encoding":
                len(data["features"]),

            "features_after_encoding":
                data["X"].shape[1],

            "numeric_features":
                len(
                    data[
                        "numeric_features"
                    ]
                ),

            "categorical_features":
                len(
                    data[
                        "categorical_features"
                    ]
                ),

            "target":
                "PM2.5_Class",

            "threshold":
                PM25_THRESHOLD
        },

        "concept_calculations":
            concept_calculations,

        "features":
            data["features"],

        "feature_importance":
            feature_importance_records,

        "prediction_table":
            prediction_table,

        "error_analysis": {

            "TN":
                selected["TN"],

            "FP":
                selected["FP"],

            "FN":
                selected["FN"],

            "TP":
                selected["TP"],

            "correct":
                selected["correct"],

            "incorrect":
                selected["incorrect"],

            "error_percentage":
                selected[
                    "error_percentage"
                ]
        },

        "model_comparison":
            model_comparison,

        "best_model": {

            "key":
                best_key,

            "name":
                all_results[
                    best_key
                ]["model_name"],

            "f1":
                all_results[
                    best_key
                ]["f1"],

            "accuracy":
                all_results[
                    best_key
                ]["accuracy"],

            "roc_auc":
                all_results[
                    best_key
                ]["roc_auc"]
        },

        "charts":
            charts,

        "regression_tree":
            regression_tree_info,

        "available_models":
            list(
                all_results.keys()
            )
    }

    # ========================================================
    # SAVE CACHE
    # ========================================================

    cache_data = {

        "cache_version":
            CACHE_VERSION,

        "all_results":
            all_results,

        "data":
            data,

        "concept_calculations":
            concept_calculations
    }

    joblib.dump(
        cache_data,
        CACHE_FILE
    )

    print(
        "\nDecision Tree Analysis completed."
    )

    print(
        f"Selected Model: "
        f"{selected['model_name']}"
    )

    print(
        f"Accuracy: "
        f"{selected['accuracy']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{selected['f1']:.4f}"
    )

    return result