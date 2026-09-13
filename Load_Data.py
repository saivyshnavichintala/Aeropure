import os
import pandas as pd


# ============================================================
# AEROPURE - DATA LOADING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Merged_PRSA_Data.csv"
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file does not exist:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    return df


def get_data_summary(path: str = DATA_PATH) -> dict:

    df = load_data(path)

    summary = {

        "n_rows": int(df.shape[0]),

        "n_cols": int(df.shape[1]),

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
        )
    }

    return summary


if __name__ == "__main__":

    try:

        df = load_data()

        print("=" * 70)
        print("AEROPURE - DATASET LOADING")
        print("=" * 70)

        print("\nDataset Path:")
        print(DATA_PATH)

        print("\nDataset Shape:")
        print(df.shape)

        print("\nNumber of Rows:")
        print(df.shape[0])

        print("\nNumber of Columns:")
        print(df.shape[1])

        print("\nColumns:")
        print(list(df.columns))

        print("\nFirst 10 Rows:")
        print(df.head(10))

        print("\nMissing Values:")
        print(df.isna().sum())

        print("\nData Types:")
        print(df.dtypes)

        print("\n" + "=" * 70)
        print("DATASET LOADED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:

        print("\nERROR:")
        print(e)