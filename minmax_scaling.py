import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


# =========================================================
# 1. LOAD DATASET
# =========================================================

file_path = r"C:\Users\saivy\OneDrive\Documents\Machine Learning\SDP\SDP\Merged_PRSA_Data.csv"

df = pd.read_csv(
    file_path,
    low_memory=False
)


print("=================================================")
print("AEROPURE - MIN-MAX SCALING")
print("=================================================")

print("\nOriginal Dataset Shape:")
print(df.shape)


# =========================================================
# 2. NUMERICAL FEATURES
# =========================================================

features = [

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
# 3. TARGET VARIABLE
# =========================================================

target = "PM2.5"


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = features + [
    target,
    "year"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    print("\nERROR!")

    print(
        "The following columns are missing:"
    )

    print(
        missing_columns
    )

    exit()


# =========================================================
# 5. REMOVE MISSING TARGET VALUES
# =========================================================

df = df.dropna(
    subset=[target]
)


print("\nDataset Shape After Removing")
print("Missing PM2.5 Values:")

print(df.shape)


# =========================================================
# 6. HANDLE MISSING FEATURE VALUES
# =========================================================

print("\nMissing Values Before Filling:")

print(
    df[features]
    .isnull()
    .sum()
)


for column in features:

    df[column] = df[column].fillna(
        df[column].median()
    )


print("\nMissing Values After Filling:")

print(
    df[features]
    .isnull()
    .sum()
)


# =========================================================
# 7. TIME-BASED TRAIN TEST SPLIT
# =========================================================
#
# Training:
#       Year < 2017
#
# Testing:
#       Year == 2017
#
# =========================================================

train_df = df[
    df["year"] < 2017
].copy()


test_df = df[
    df["year"] == 2017
].copy()


print("\n=================================================")
print("TRAINING AND TESTING DATA")
print("=================================================")

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)


# =========================================================
# 8. DISPLAY DATA BEFORE SCALING
# =========================================================

print("\n=================================================")
print("TRAINING DATA BEFORE MIN-MAX SCALING")
print("=================================================")

print(
    train_df[
        features
    ].head(5)
)


print("\n=================================================")
print("TESTING DATA BEFORE MIN-MAX SCALING")
print("=================================================")

print(
    test_df[
        features
    ].head(5)
)


# =========================================================
# 9. CREATE MIN-MAX SCALER
# =========================================================

scaler = MinMaxScaler()


# =========================================================
# 10. FIT ONLY ON TRAINING DATA
# =========================================================

train_df[features] = scaler.fit_transform(
    train_df[features]
)


# =========================================================
# 11. TRANSFORM TESTING DATA
# =========================================================

test_df[features] = scaler.transform(
    test_df[features]
)


# =========================================================
# 12. DISPLAY SCALED TRAINING DATA
# =========================================================

print("\n=================================================")
print("SCALED TRAINING DATA")
print("=================================================")

print(
    train_df[
        features
    ].head(5)
)


# =========================================================
# 13. DISPLAY SCALED TESTING DATA
# =========================================================

print("\n=================================================")
print("SCALED TESTING DATA")
print("=================================================")

print(
    test_df[
        features
    ].head(5)
)


# =========================================================
# 14. CHECK TRAINING MINIMUM
# =========================================================

print("\n=================================================")
print("TRAINING DATA MINIMUM")
print("=================================================")

print(
    train_df[
        features
    ].min()
)


# =========================================================
# 15. CHECK TRAINING MAXIMUM
# =========================================================

print("\n=================================================")
print("TRAINING DATA MAXIMUM")
print("=================================================")

print(
    train_df[
        features
    ].max()
)


# =========================================================
# 16. CHECK TESTING MINIMUM
# =========================================================

print("\n=================================================")
print("TESTING DATA MINIMUM")
print("=================================================")

print(
    test_df[
        features
    ].min()
)


# =========================================================
# 17. CHECK TESTING MAXIMUM
# =========================================================

print("\n=================================================")
print("TESTING DATA MAXIMUM")
print("=================================================")

print(
    test_df[
        features
    ].max()
)


# =========================================================
# 18. DISPLAY TARGET VARIABLE
# =========================================================

print("\n=================================================")
print("TARGET VARIABLE - PM2.5")
print("=================================================")

print(
    train_df[
        [target]
    ].head(5)
)


# =========================================================
# 19. SAVE PROCESSED DATA
# =========================================================

base_directory = os.path.dirname(
    os.path.abspath(__file__)
)


output_directory = os.path.join(
    base_directory,
    "processed_data"
)


os.makedirs(
    output_directory,
    exist_ok=True
)


# =========================================================
# 20. TRAINING FILE
# =========================================================

train_file = os.path.join(
    output_directory,
    "AeroPure_MinMax_Train.csv"
)


# =========================================================
# 21. TESTING FILE
# =========================================================

test_file = os.path.join(
    output_directory,
    "AeroPure_MinMax_Test.csv"
)


# =========================================================
# 22. SAVE FILES
# =========================================================

train_df.to_csv(
    train_file,
    index=False
)


test_df.to_csv(
    test_file,
    index=False
)


# =========================================================
# 23. FINAL MESSAGE
# =========================================================

print("\n=================================================")
print("MIN-MAX SCALING COMPLETED SUCCESSFULLY")
print("=================================================")

print("\nTraining File:")
print(train_file)

print("\nTesting File:")
print(test_file)

print("\nTarget Variable:")
print(target)

print("\nFeatures Scaled:")

for feature in features:

    print(
        " -",
        feature
    )

print("\nScaling Formula:")

print(
    "X_scaled = (X - X_min) / (X_max - X_min)"
)

print("\nThe scaler was FIT only on the training data.")

print(
    "The same scaler was TRANSFORMED on the testing data."
)