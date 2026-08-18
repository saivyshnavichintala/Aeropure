import pandas as pd

# =========================================================
# LOAD DATASET
# =========================================================

file_path = r"C:\Users\saivy\OneDrive\Documents\Machine Learning\SDP\SDP\dataset\Beijing_Air_Quality_Merged.csv"

df = pd.read_csv(file_path)

print("Original Dataset Shape:")
print(df.shape)


# =========================================================
# FEATURE
# =========================================================

feature = "PM2.5"

print("\nOriginal Statistics:")
print(df[feature].describe())


# =========================================================
# 1. CALCULATE IQR
# =========================================================

Q1 = df[feature].quantile(0.25)

Q3 = df[feature].quantile(0.75)

IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR

upper_fence = Q3 + 1.5 * IQR


print("\nLower Fence:")
print(lower_fence)

print("\nUpper Fence:")
print(upper_fence)


print("\nMinimum BEFORE clipping:")
print(df[feature].min())

print("\nMaximum BEFORE clipping:")
print(df[feature].max())


# =========================================================
# 2. FIND OUTLIERS
# =========================================================

outliers = df[
    (df[feature] < lower_fence) |
    (df[feature] > upper_fence)
]


print("\nNumber of Outliers:")
print(len(outliers))


# =========================================================
# 3. CLIP OUTLIERS
# =========================================================

df["PM2.5_Clipped"] = df[feature].clip(
    lower=lower_fence,
    upper=upper_fence
)


print("\nMinimum AFTER clipping:")
print(df["PM2.5_Clipped"].min())

print("\nMaximum AFTER clipping:")
print(df["PM2.5_Clipped"].max())


# =========================================================
# 4. DISPLAY RESULT
# =========================================================

print("\nOriginal and Clipped Data:")

print(
    df[
        [
            "PM2.5",
            "PM2.5_Clipped"
        ]
    ].head(10)
)