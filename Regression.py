import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)


def find_pm25_column(df):

    possible_names = [
        "PM2.5",
        "PM2_5",
        "pm2.5",
        "pm25",
        "PM25"
    ]

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def run_regression(df):

    results = {}

    pm25_column = find_pm25_column(df)

    # ------------------------------------------------
    # If PM2.5 doesn't exist
    # ------------------------------------------------

    if pm25_column is None:

        results["error"] = (
            "PM2.5 column was not found in the dataset."
        )

        return results

    # ------------------------------------------------
    # Prepare numerical data
    # ------------------------------------------------

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if pm25_column not in numerical_columns:

        results["error"] = (
            "PM2.5 must be a numerical column."
        )

        return results

    feature_columns = [
        column
        for column in numerical_columns
        if column != pm25_column
    ]

    if len(feature_columns) == 0:

        results["error"] = (
            "No numerical feature columns available."
        )

        return results

    data = df[
        feature_columns + [pm25_column]
    ].dropna()

    X = data[feature_columns]
    y = data[pm25_column]

    # =================================================
    # LINEAR REGRESSION
    # =================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    predictions = linear_model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results["linear_mse"] = round(mse, 3)
    results["linear_r2"] = round(r2, 3)

    # =================================================
    # LOGISTIC REGRESSION
    # =================================================

    # Create binary air-quality class
    # 0 = Lower PM2.5
    # 1 = Higher PM2.5

    threshold = y.median()

    y_class = (
        y >= threshold
    ).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    logistic_model = LogisticRegression(
        max_iter=1000
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    predictions = logistic_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results["logistic_accuracy"] = round(
        accuracy,
        3
    )

    results["threshold"] = round(
        threshold,
        2
    )

    results["features"] = feature_columns

    return results