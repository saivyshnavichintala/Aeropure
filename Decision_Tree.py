from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
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


def run_decision_tree(df):

    results = {}

    pm25_column = find_pm25_column(df)

    if pm25_column is None:

        results["error"] = (
            "PM2.5 column was not found."
        )

        return results

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    feature_columns = [
        column
        for column in numerical_columns
        if column != pm25_column
    ]

    if len(feature_columns) == 0:

        results["error"] = (
            "No numerical features available."
        )

        return results

    data = df[
        feature_columns + [pm25_column]
    ].dropna()

    X = data[feature_columns]
    y = data[pm25_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =================================================
    # BEFORE PRUNING
    # =================================================

    tree_before = DecisionTreeRegressor(
        random_state=42
    )

    tree_before.fit(
        X_train,
        y_train
    )

    predictions_before = tree_before.predict(
        X_test
    )

    mse_before = mean_squared_error(
        y_test,
        predictions_before
    )

    r2_before = r2_score(
        y_test,
        predictions_before
    )

    # =================================================
    # AFTER PRUNING
    # =================================================

    tree_after = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    tree_after.fit(
        X_train,
        y_train
    )

    predictions_after = tree_after.predict(
        X_test
    )

    mse_after = mean_squared_error(
        y_test,
        predictions_after
    )

    r2_after = r2_score(
        y_test,
        predictions_after
    )

    results["mse_before"] = round(
        mse_before,
        3
    )

    results["r2_before"] = round(
        r2_before,
        3
    )

    results["mse_after"] = round(
        mse_after,
        3
    )

    results["r2_after"] = round(
        r2_after,
        3
    )

    results["depth_before"] = (
        tree_before.get_depth()
    )

    results["depth_after"] = (
        tree_after.get_depth()
    )

    return results