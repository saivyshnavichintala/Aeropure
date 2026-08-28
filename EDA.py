import pandas as pd


def run_eda(df):

    results = {}

    # Basic information
    results["rows"] = len(df)
    results["columns"] = len(df.columns)

    # Missing values
    results["missing"] = int(
        df.isnull().sum().sum()
    )

    # Duplicate values
    results["duplicates"] = int(
        df.duplicated().sum()
    )

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    results["numerical_columns"] = numerical_columns

    # Statistical summary
    if numerical_columns:

        results["statistics"] = (
            df[numerical_columns]
            .describe()
            .round(2)
            .to_html(
                classes="data-table"
            )
        )

    else:

        results["statistics"] = (
            "<p>No numerical columns found.</p>"
        )

    # Correlation
    if len(numerical_columns) > 1:

        correlation = (
            df[numerical_columns]
            .corr()
            .round(2)
        )

        results["correlation"] = (
            correlation.to_html(
                classes="data-table"
            )
        )

    else:

        results["correlation"] = (
            "<p>Not enough numerical columns.</p>"
        )

    return results