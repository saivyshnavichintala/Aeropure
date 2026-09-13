# ============================================================
# DECISION TREE AND ENSEMBLE LEARNING
# ============================================================
#
# Topics:
# 1. Decision Tree Basics
# 2. Entropy
# 3. Information Gain
# 4. Gini Index
# 5. Gain Ratio
# 6. ID3
# 7. C4.5
# 8. CART
# 9. Tree Construction
# 10. Overfitting
# 11. Pruning
# 12. Regression Tree
# 13. Random Forest
# 14. AdaBoost
# 15. Gradient Boosting
# 16. XGBoost
# 17. LightGBM
#
# ============================================================

import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text
)

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error
)

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)


# ============================================================
# 1. DECISION TREE BASICS
# ============================================================

def decision_tree_basics():

    print("\n========== DECISION TREE BASICS ==========")

    data = {
        "CGPA": [6.5, 7.2, 8.1, 9.0, 6.8, 8.5, 7.0, 9.2],
        "Attendance": [60, 70, 85, 95, 65, 90, 72, 98],
        "Placed": [0, 0, 1, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    X = df[["CGPA", "Attendance"]]
    y = df["Placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print("Predictions:", prediction)
    print("Actual:", y_test.values)

    print(
        "Accuracy:",
        accuracy_score(y_test, prediction)
    )


# ============================================================
# 2. ENTROPY
# ============================================================

def entropy(data):

    if len(data) == 0:
        return 0

    counts = {}

    for value in data:

        if value not in counts:
            counts[value] = 0

        counts[value] += 1

    result = 0

    total = len(data)

    for count in counts.values():

        probability = count / total

        if probability > 0:

            result -= (
                probability *
                math.log2(probability)
            )

    return result


def entropy_demo():

    print("\n========== ENTROPY ==========")

    data1 = [1, 1, 1, 1]

    data2 = [1, 1, 0, 0]

    print("Dataset 1:", data1)

    print(
        "Entropy:",
        entropy(data1)
    )

    print("\nDataset 2:", data2)

    print(
        "Entropy:",
        entropy(data2)
    )


# ============================================================
# 3. INFORMATION GAIN
# ============================================================

def information_gain(parent, groups):

    parent_entropy = entropy(parent)

    weighted_child_entropy = 0

    for group in groups:

        weight = len(group) / len(parent)

        weighted_child_entropy += (
            weight * entropy(group)
        )

    gain = (
        parent_entropy -
        weighted_child_entropy
    )

    return gain


def information_gain_demo():

    print("\n========== INFORMATION GAIN ==========")

    parent = [
        1, 1, 1,
        0, 0, 0
    ]

    left = [
        1, 1, 1
    ]

    right = [
        0, 0, 0
    ]

    gain = information_gain(
        parent,
        [left, right]
    )

    print("Parent:", parent)

    print("Left:", left)

    print("Right:", right)

    print(
        "Information Gain:",
        gain
    )


# ============================================================
# 4. GINI INDEX
# ============================================================

def gini(data):

    if len(data) == 0:
        return 0

    counts = {}

    for value in data:

        if value not in counts:
            counts[value] = 0

        counts[value] += 1

    total = len(data)

    result = 1

    for count in counts.values():

        probability = count / total

        result -= probability ** 2

    return result


def gini_demo():

    print("\n========== GINI INDEX ==========")

    data1 = [1, 1, 1, 1]

    data2 = [1, 1, 0, 0]

    print(
        "Gini of",
        data1,
        "=",
        gini(data1)
    )

    print(
        "Gini of",
        data2,
        "=",
        gini(data2)
    )


# ============================================================
# 5. GAIN RATIO
# ============================================================

def gain_ratio(parent, groups):

    if len(parent) == 0:
        return 0

    parent_entropy = entropy(parent)

    weighted_entropy = 0

    split_information = 0

    for group in groups:

        if len(group) == 0:
            continue

        weight = len(group) / len(parent)

        weighted_entropy += (
            weight * entropy(group)
        )

        split_information -= (
            weight * math.log2(weight)
        )

    information_gain_value = (
        parent_entropy -
        weighted_entropy
    )

    if split_information == 0:
        return 0

    return (
        information_gain_value /
        split_information
    )


def gain_ratio_demo():

    print("\n========== GAIN RATIO ==========")

    parent = [
        1, 1, 1,
        0, 0, 0
    ]

    group1 = [1, 1, 1]

    group2 = [0, 0, 0]

    result = gain_ratio(
        parent,
        [group1, group2]
    )

    print(
        "Gain Ratio:",
        result
    )


# ============================================================
# 6. ID3
# ============================================================
#
# ID3 uses:
#
# Entropy
# +
# Information Gain
#
# ============================================================

def id3_demo():

    print("\n========== ID3 ==========")

    data = {

        "Outlook": [
            "Sunny",
            "Sunny",
            "Overcast",
            "Rain",
            "Rain",
            "Rain",
            "Overcast",
            "Sunny"
        ],

        "Temperature": [
            "Hot",
            "Hot",
            "Hot",
            "Mild",
            "Cool",
            "Cool",
            "Cool",
            "Mild"
        ],

        "Play": [
            0, 0, 1, 1,
            1, 0, 1, 0
        ]
    }

    df = pd.DataFrame(data)

    X = pd.get_dummies(
        df.drop("Play", axis=1),
        dtype=int
    )

    y = df["Play"]

    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    )

    model.fit(X, y)

    print(
        export_text(
            model,
            feature_names=list(X.columns)
        )
    )


# ============================================================
# 7. C4.5
# ============================================================
#
# C4.5 uses Gain Ratio.
#
# scikit-learn does not directly implement C4.5.
#
# ============================================================

def c45_demo():

    print("\n========== C4.5 ==========")

    parent = [
        1, 1, 1,
        0, 0, 0
    ]

    group1 = [
        1, 1, 1
    ]

    group2 = [
        0, 0, 0
    ]

    result = gain_ratio(
        parent,
        [group1, group2]
    )

    print(
        "C4.5 uses Gain Ratio."
    )

    print(
        "Example Gain Ratio:",
        result
    )


# ============================================================
# 8. CART
# ============================================================
#
# CART Classification uses Gini Index.
#
# ============================================================

def cart_demo():

    print("\n========== CART ==========")

    data = {

        "CGPA": [
            6.2,
            6.8,
            7.1,
            7.8,
            8.2,
            8.7,
            9.1,
            9.5
        ],

        "Attendance": [
            55,
            60,
            68,
            72,
            80,
            85,
            90,
            95
        ],

        "Placed": [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1
        ]
    }

    df = pd.DataFrame(data)

    X = df[
        [
            "CGPA",
            "Attendance"
        ]
    ]

    y = df["Placed"]

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

    model.fit(X, y)

    print(
        export_text(
            model,
            feature_names=[
                "CGPA",
                "Attendance"
            ]
        )
    )


# ============================================================
# 9. TREE CONSTRUCTION
# ============================================================

def tree_construction():

    print("\n========== TREE CONSTRUCTION ==========")

    data = {

        "CGPA": [
            6.2,
            6.8,
            7.1,
            7.8,
            8.2,
            8.7,
            9.1,
            9.5
        ],

        "Attendance": [
            55,
            60,
            68,
            72,
            80,
            85,
            90,
            95
        ],

        "Placed": [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1
        ]
    }

    df = pd.DataFrame(data)

    X = df[
        [
            "CGPA",
            "Attendance"
        ]
    ]

    y = df["Placed"]

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    print(
        "Decision Tree constructed."
    )

    plt.figure(
        figsize=(12, 7)
    )

    plot_tree(
        model,

        feature_names=[
            "CGPA",
            "Attendance"
        ],

        class_names=[
            "Not Placed",
            "Placed"
        ],

        filled=True
    )

    plt.title(
        "Decision Tree"
    )

    plt.show()


# ============================================================
# 10. OVERFITTING
# ============================================================

def overfitting_demo():

    print("\n========== OVERFITTING ==========")

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    shallow = DecisionTreeClassifier(
        max_depth=2,
        random_state=42
    )

    deep = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    shallow.fit(
        X_train,
        y_train
    )

    deep.fit(
        X_train,
        y_train
    )

    shallow_train = accuracy_score(
        y_train,
        shallow.predict(X_train)
    )

    shallow_test = accuracy_score(
        y_test,
        shallow.predict(X_test)
    )

    deep_train = accuracy_score(
        y_train,
        deep.predict(X_train)
    )

    deep_test = accuracy_score(
        y_test,
        deep.predict(X_test)
    )

    print("\nShallow Tree")

    print(
        "Training Accuracy:",
        shallow_train
    )

    print(
        "Testing Accuracy:",
        shallow_test
    )

    print("\nDeep Tree")

    print(
        "Training Accuracy:",
        deep_train
    )

    print(
        "Testing Accuracy:",
        deep_test
    )


# ============================================================
# 11. PRUNING
# ============================================================

def pruning_demo():

    print("\n========== PRUNING ==========")

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = DecisionTreeClassifier(

        criterion="gini",

        max_depth=3,

        min_samples_split=4,

        min_samples_leaf=2,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Pruned Tree Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


# ============================================================
# 12. REGRESSION TREE
# ============================================================

def regression_tree():

    print("\n========== REGRESSION TREE ==========")

    data = {

        "Hours": [
            1, 2, 3, 4,
            5, 6, 7, 8
        ],

        "Marks": [
            20, 30, 40, 50,
            60, 70, 80, 90
        ]
    }

    df = pd.DataFrame(data)

    X = df[["Hours"]]

    y = df["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Predictions:",
        prediction
    )

    print(
        "Actual:",
        y_test.values
    )

    print(
        "Mean Squared Error:",
        mean_squared_error(
            y_test,
            prediction
        )
    )


# ============================================================
# 13. RANDOM FOREST
# ============================================================

def random_forest():

    print("\n========== RANDOM FOREST ==========")

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


# ============================================================
# 14. ADABOOST
# ============================================================

def adaboost():

    print("\n========== ADABOOST ==========")

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = AdaBoostClassifier(

        n_estimators=100,

        learning_rate=0.5,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


# ============================================================
# 15. GRADIENT BOOSTING
# ============================================================

def gradient_boosting():

    print("\n========== GRADIENT BOOSTING ==========")

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = GradientBoostingClassifier(

        n_estimators=100,

        learning_rate=0.1,

        max_depth=3,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


# ============================================================
# 16. XGBOOST
# ============================================================

def xgboost_demo():

    print("\n========== XGBOOST ==========")

    try:

        from xgboost import XGBClassifier

    except ImportError:

        print(
            "XGBoost is not installed."
        )

        print(
            "Run: pip install xgboost"
        )

        return

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(

        n_estimators=100,

        learning_rate=0.1,

        max_depth=3,

        random_state=42,

        eval_metric="mlogloss"
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


# ============================================================
# 17. LIGHTGBM
# ============================================================

def lightgbm_demo():

    print("\n========== LIGHTGBM ==========")

    try:

        from lightgbm import LGBMClassifier

    except ImportError:

        print(
            "LightGBM is not installed."
        )

        print(
            "Run: pip install lightgbm"
        )

        return

    data = load_iris()

    X = data.data

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = LGBMClassifier(

        n_estimators=100,

        learning_rate=0.1,

        num_leaves=31,

        max_depth=-1,

        random_state=42,

        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print("\n")
        print("=" * 60)

        print(
            "DECISION TREE + ENSEMBLE LEARNING"
        )

        print("=" * 60)

        print("1.  Decision Tree Basics")
        print("2.  Entropy")
        print("3.  Information Gain")
        print("4.  Gini Index")
        print("5.  Gain Ratio")
        print("6.  ID3")
        print("7.  C4.5")
        print("8.  CART")
        print("9.  Tree Construction")
        print("10. Overfitting")
        print("11. Pruning")
        print("12. Regression Tree")
        print("13. Random Forest")
        print("14. AdaBoost")
        print("15. Gradient Boosting")
        print("16. XGBoost")
        print("17. LightGBM")
        print("0.  Exit")

        choice = input(
            "\nEnter your choice: "
        )

        if choice == "1":
            decision_tree_basics()

        elif choice == "2":
            entropy_demo()

        elif choice == "3":
            information_gain_demo()

        elif choice == "4":
            gini_demo()

        elif choice == "5":
            gain_ratio_demo()

        elif choice == "6":
            id3_demo()

        elif choice == "7":
            c45_demo()

        elif choice == "8":
            cart_demo()

        elif choice == "9":
            tree_construction()

        elif choice == "10":
            overfitting_demo()

        elif choice == "11":
            pruning_demo()

        elif choice == "12":
            regression_tree()

        elif choice == "13":
            random_forest()

        elif choice == "14":
            adaboost()

        elif choice == "15":
            gradient_boosting()

        elif choice == "16":
            xgboost_demo()

        elif choice == "17":
            lightgbm_demo()

        elif choice == "0":
            print("Program ended.")
            break

        else:
            print(
                "Invalid choice. Try again."
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()