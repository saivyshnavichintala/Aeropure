
import os
import pandas as pd


# =====================================================
# DATASET PATH
# =====================================================

DATA_PATH = r"C:\Users\chint\PycharmProjects\Aeropure\Merged_PRSA_Data.csv"


# =====================================================
# LOAD DATA
# =====================================================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    df = pd.read_csv(path)

    # Remove duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# =====================================================
# DATA SUMMARY
# =====================================================

def get_data_summary(path: str = DATA_PATH) -> dict:

    df = load_data(path)

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict(
            orient="records"
        ),
    }

    return summary


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    df = load_data()

    print("Dataset loaded successfully!")

    print("Rows:", df.shape[0])

    print("Columns:", df.shape[1])

    print("\nColumn names:")

    print(df.columns.tolist())

    print("\nFirst 10 rows:")

    print(df.head(10))

