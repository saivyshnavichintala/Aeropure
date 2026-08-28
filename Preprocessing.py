import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def preprocess_data(df):

    data = df.copy()

    results = {}

    # ---------------------------------------------
    # Original information
    # ---------------------------------------------

    results["original_rows"] = len(data)
    results["original_columns"] = len(data.columns)

    # ---------------------------------------------
    # Missing values
    # ---------------------------------------------

    missing_before = int(
        data.isnull().sum().sum()
    )

    results["missing_before"] = missing_before

    # Fill numerical missing values
    numerical_columns = data.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        data[column] = data[column].fillna(
            data[column].median()
        )

    # Fill categorical missing values
    categorical_columns = data.select_dtypes(
        exclude="number"
    ).columns

    for column in categorical_columns:

        if not data[column].mode().empty:

            data[column] = data[column].fillna(
                data[column].mode()[0]
            )

    missing_after = int(
        data.isnull().sum().sum()
    )

    results["missing_after"] = missing_after

    # ---------------------------------------------
    # IQR Outlier Detection
    # ---------------------------------------------

    outlier_count = 0

    for column in numerical_columns:

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = (
            (data[column] < lower) |
            (data[column] > upper)
        )

        outlier_count += int(outliers.sum())

        # IQR clipping
        data[column] = data[column].clip(
            lower,
            upper
        )

    results["outliers"] = outlier_count

    # ---------------------------------------------
    # Min-Max Scaling
    # ---------------------------------------------

    if len(numerical_columns) > 0:

        scaler = MinMaxScaler()

        data[numerical_columns] = scaler.fit_transform(
            data[numerical_columns]
        )

    results["scaled_columns"] = list(
        numerical_columns
    )

    results["processed_data"] = (
        data.head(20)
        .round(3)
        .to_html(
            classes="data-table",
            index=False
        )
    )

    return results