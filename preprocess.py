import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# =========================================================
# PREPROCESSING FUNCTION
# =========================================================

def run_preprocessing():

    # =====================================================
    # LOAD DATASET
    # =====================================================

    file_path = (
        r"C:\Users\saivy\OneDrive\Documents\Machine Learning\SDP"
        r"\SDP\Merged_PRSA_Data.csv"
    )

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    original_rows = len(df)
    original_columns = len(df.columns)


    # =====================================================
    # NUMERICAL FEATURES
    # =====================================================

    num_cols = [
        "PM2.5",
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


    # =====================================================
    # CHECK REQUIRED COLUMNS
    # =====================================================

    missing_columns = [
        col for col in num_cols
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The following columns are missing from the dataset: "
            + ", ".join(missing_columns)
        )


    # =====================================================
    # MISSING VALUES BEFORE
    # =====================================================

    missing_before = (
        df[num_cols]
        .isnull()
        .sum()
        .to_dict()
    )


    # =====================================================
    # HANDLE MISSING VALUES
    # MEDIAN IMPUTATION
    # =====================================================

    for col in num_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )


    # =====================================================
    # MISSING VALUES AFTER
    # =====================================================

    missing_after = (
        df[num_cols]
        .isnull()
        .sum()
        .to_dict()
    )


    # =====================================================
    # OUTLIER DETECTION
    # IQR METHOD
    # =====================================================

    outlier_results = {}

    for feature in num_cols:

        Q1 = df[feature].quantile(0.25)

        Q3 = df[feature].quantile(0.75)

        IQR = Q3 - Q1

        lower_fence = Q1 - (1.5 * IQR)

        upper_fence = Q3 + (1.5 * IQR)


        # Find outliers

        outliers = df[
            (df[feature] < lower_fence)
            |
            (df[feature] > upper_fence)
        ]


        outlier_results[feature] = {

            "Q1": round(Q1, 4),

            "Q3": round(Q3, 4),

            "IQR": round(IQR, 4),

            "lower_fence":
                round(lower_fence, 4),

            "upper_fence":
                round(upper_fence, 4),

            "outliers":
                len(outliers)

        }


        # =================================================
        # CLIP OUTLIERS
        # =================================================

        df[feature] = df[feature].clip(
            lower=lower_fence,
            upper=upper_fence
        )


    # =====================================================
    # CREATE DATETIME
    # =====================================================

    if all(
        col in df.columns
        for col in ["year", "month", "day", "hour"]
    ):

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


    # =====================================================
    # SORT DATA
    # =====================================================

    if "station" in df.columns and "datetime" in df.columns:

        df = df.sort_values(
            [
                "station",
                "datetime"
            ]
        )


    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    if "year" in df.columns:

        train_df = df[
            df["year"] < 2017
        ].copy()

        test_df = df[
            df["year"] == 2017
        ].copy()

    else:

        # Fallback if year column does not exist

        split_index = int(
            len(df) * 0.8
        )

        train_df = df.iloc[
            :split_index
        ].copy()

        test_df = df.iloc[
            split_index:
        ].copy()


    # =====================================================
    # SAVE DATA BEFORE SCALING
    # FOR DISPLAY
    # =====================================================

    scaling_before = (
        train_df[num_cols]
        .head(5)
        .round(4)
        .to_dict(orient="records")
    )


    testing_before = (
        test_df[num_cols]
        .head(5)
        .round(4)
        .to_dict(orient="records")
    )


    # =====================================================
    # MIN-MAX SCALING
    # =====================================================

    scaler = MinMaxScaler()


    # =====================================================
    # FIT ONLY ON TRAINING DATA
    # =====================================================

    train_df[num_cols] = scaler.fit_transform(
        train_df[num_cols]
    )


    # =====================================================
    # TRANSFORM TEST DATA
    # =====================================================

    test_df[num_cols] = scaler.transform(
        test_df[num_cols]
    )


    # =====================================================
    # SCALED TRAINING DATA
    # =====================================================

    scaling_after = (
        train_df[num_cols]
        .head(5)
        .round(4)
        .to_dict(orient="records")
    )


    # =====================================================
    # SCALED TESTING DATA
    # =====================================================

    testing_after = (
        test_df[num_cols]
        .head(5)
        .round(4)
        .to_dict(orient="records")
    )


    # =====================================================
    # MINIMUM VALUES AFTER SCALING
    # =====================================================

    scaled_minimum = (
        train_df[num_cols]
        .min()
        .round(4)
        .to_dict()
    )


    # =====================================================
    # MAXIMUM VALUES AFTER SCALING
    # =====================================================

    scaled_maximum = (
        train_df[num_cols]
        .max()
        .round(4)
        .to_dict()
    )


    # =====================================================
    # SCALER INFORMATION
    # =====================================================

    scaler_information = {}

    for i, feature in enumerate(num_cols):

        scaler_information[feature] = {

            "original_min":
                round(
                    scaler.data_min_[i],
                    4
                ),

            "original_max":
                round(
                    scaler.data_max_[i],
                    4
                )

        }


    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    output_directory = os.path.join(
        os.path.dirname(__file__),
        "processed_data"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )


    # =====================================================
    # OUTPUT FILE PATHS
    # =====================================================

    train_path = os.path.join(
        output_directory,
        "AeroPure_Train.csv"
    )

    test_path = os.path.join(
        output_directory,
        "AeroPure_Test.csv"
    )


    # =====================================================
    # SAVE PROCESSED DATA
    # =====================================================

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )


    # =====================================================
    # TOTAL OUTLIERS
    # =====================================================

    total_outliers = sum(
        item["outliers"]
        for item in outlier_results.values()
    )


    # =====================================================
    # RETURN ALL RESULTS
    # =====================================================

    return {

        # Dataset information

        "original_rows":
            original_rows,

        "original_columns":
            original_columns,

        "train_rows":
            len(train_df),

        "test_rows":
            len(test_df),

        # Features

        "features":
            num_cols,

        # Missing values

        "missing_before":
            missing_before,

        "missing_after":
            missing_after,

        # Outliers

        "outliers":
            outlier_results,

        "total_outliers":
            total_outliers,

        # Min-Max Scaling

        "scaling_before":
            scaling_before,

        "testing_before":
            testing_before,

        "scaling_after":
            scaling_after,

        "testing_after":
            testing_after,

        "scaled_minimum":
            scaled_minimum,

        "scaled_maximum":
            scaled_maximum,

        "scaler_information":
            scaler_information,

        # Files

        "train_file":
            train_path,

        "test_file":
            test_path
    }