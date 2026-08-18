import pandas as pd

# =========================================================
# LOAD DATASET
# =========================================================

file_path = r"C:\Users\saivy\OneDrive\Documents\Machine Learning\SDP\SDP\dataset\Beijing_Air_Quality_Merged.csv"

df = pd.read_csv(file_path)

print("Original Dataset Shape:")
print(df.shape)


# =========================================================
# NUMERICAL FEATURES
# =========================================================

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


# =========================================================
# FIND AND CLIP OUTLIERS
# =========================================================

for feature in num_cols:

    print("\n========================================")
    print("Feature:", feature)
    print("========================================")

    Q1 = df[feature].quantile(0.25)

    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR

    upper_fence = Q3 + 1.5 * IQR

    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)

    print("Lower Fence:", lower_fence)
    print("Upper Fence:", upper_fence)

    # -----------------------------------------------------
    # FIND OUTLIERS
    # -----------------------------------------------------

    outliers = df[
        (df[feature] < lower_fence) |
        (df[feature] > upper_fence)
    ]

    print("Number of Outliers:", len(outliers))

    # -----------------------------------------------------
    # CLIP OUTLIERS
    # -----------------------------------------------------

    df[feature + "_Clipped"] = df[feature].clip(
        lower=lower_fence,
        upper=upper_fence
    )

    print(
        "Minimum AFTER clipping:",
        df[feature + "_Clipped"].min()
    )

    print(
        "Maximum AFTER clipping:",
        df[feature + "_Clipped"].max()
    )


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\nFinal Dataset:")
print(df.head())


# =========================================================
# SAVE DATASET
# =========================================================

output_path = r"C:\Users\saivy\OneDrive\Documents\Machine Learning\SDP\SDP\dataset\AirQuality_Outliers_Clipped.csv"

df.to_csv(output_path, index=False)

print("\nDataset saved successfully!")
print(output_path)